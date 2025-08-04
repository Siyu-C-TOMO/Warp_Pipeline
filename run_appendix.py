#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from pathlib import Path

import config as cfg

from pipeline_utils import run_command

def reconstruction(log_file_path: Path):
    """Runs the final reconstruction and packaging stage for Windows compatibility."""
    logging.info("Starting final reconstruction and packaging stage...")

    win_dir = Path("forWindows_frames")
    win_dir.mkdir(exist_ok=True)
    logging.info(f"{win_dir.resolve()} is ready for packaging.")

    logging.info("Running WarpTools ts_reconstruct...")
    env = os.environ.copy()
    env['WARP_FORCE_MRC_FLOAT32'] = '1'

    cmd_reconstruct = [
        "WarpTools", "ts_reconstruct",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--device_list", str(cfg.gpu_devices[0]),
        "--perdevice", str(cfg.jobs_per_gpu)
        #"--input_data", "tomostar/L2_G1_ts_007.tomostar"
    ]
    run_command(cmd_reconstruct, log_file_path, env=env)

    logging.info(f"Linking result files into {win_dir}...")
    
    warp_frameseries_dir = Path("warp_frameseries")
    warp_tiltseries_dir = Path("warp_tiltseries")
    tomostar_dir = Path("tomostar")

    link_pairs = {
        "average": warp_frameseries_dir / "average",
        "reconstruction": warp_tiltseries_dir / "reconstruction",
    }
    for link_name, target_path in link_pairs.items():
        dest_link = win_dir / link_name
        if not dest_link.exists() and target_path.exists():
            # Create relative symlink
            relative_target = os.path.relpath(target_path.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)

    for xml_file in warp_tiltseries_dir.glob("*.xml"):
        dest_link = win_dir / xml_file.name
        if not dest_link.exists():
            relative_target = os.path.relpath(xml_file.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)
            
    for star_file in tomostar_dir.glob("*.tomostar"):
        dest_link = win_dir / star_file.name
        if not dest_link.exists():
            relative_target = os.path.relpath(star_file.resolve(), win_dir.resolve())
            dest_link.symlink_to(relative_target)

    logging.info("Reconstruction and packaging stage completed.")

def isonet(log_file_path: Path):
    """Run the ISONet stage of the pipeline."""

    list_file = Path('ribo_list.txt')
    if not list_file.exists():
        logging.error(f"tomogram list file does not exist in the current directory: {list_file.resolve()}")
        sys.exit(1)
    
    tomo_list = [line.split()[0] for line in list_file.read_text().strip().splitlines()]
    logging.info(f"Found {len(tomo_list)} tomograms in the list.")

    tomo_folder = "tomoset"
    isonet_dir = Path("isonet")
    Path(isonet_dir / tomo_folder).mkdir(parents=True, exist_ok=True)
    Path(isonet_dir / "logs").mkdir(parents=True, exist_ok=True)

    for tomo in tomo_list:
        source = Path(f"warp_tiltseries/tiltstack/{tomo}").resolve()
        target = isonet_dir / tomo_folder / f"{tomo}.mrc"

        source_file = list(source.glob("*dev.mrc"))[0]
        if not source_file:
            logging.warning(f"No _dev.mrc file found for tomogram {tomo}")
            continue

        if not target.exists():
            target.symlink_to(source_file)

    pixel_size = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    commands = [
        f"isonet.py prepare_star {tomo_folder} --output_star tomogram.star --pixel_size {pixel_size} --number_subtomos 40",
        "isonet.py make_mask tomogram.star --mask_folder mask --density_percentage 40 --std_percentage 40 --z_crop 0.14",
        "isonet.py extract tomogram.star --cube_size 64",
        "isonet.py refine subtomo.star --gpuID 0,1,2,3,4,5,6,7 --iterations 30 --noise_level 0.1,0.15,0.2,0.25 --noise_start_iter 10,15,20,25",
        "isonet.py predict tomogram.star ./results/model_iter30.h5 --gpuID 0,1,2,3,4,5,6,7 --cube_size 64 --crop_size 96",
    ]

    for cmd in commands:
        logging.info(f"--- Starting ISONet step: {cmd.split()[0]} ---")
        run_command(cmd, log_file_path, cwd=isonet_dir, module_load="isonet")
    
    logging.info("--- All ISONet steps completed successfully. ---")

def cryolo(log_file_path: Path):
    """Run the Cryolo stage of the pipeline."""
    cryolo_dir = Path("cryolo")
    if not cryolo_dir.exists():
        logging.error(f"No cryolo directory. Run star-handler process-relion2cryolo first.")
        sys.exit(1)

    cryolo_ad = "/software/repo/rhel9/cryolo/1.9.4/bin"
    cmd_ini = f"'{cryolo_ad}/python3.8' -u '{cryolo_ad}/cryolo_gui.py' --ignore-gooey"
    thre = 0.25
    connect_min = 5
    output_dir = Path("expand10_0p25_5")
    commands = [
        f"{cmd_ini} config --train_image_folder '3DTM_pre/tomograms' --train_annot_folder '3DTM_pre/CBOX' --saved_weights_name 'cryolo_model_fromRelion_expand10.h5' -a 'PhosaurusNet' --input_size 1024 -nm 'STANDARD' --num_patches '1' --overlap_patches '200' --filtered_output 'filtered_tmp/' -f 'LOWPASS' --low_pass_cutoff '0.1' --janni_overlap '24' --janni_batches '3' --train_times '10' --batch_size '4' --learning_rate '0.0001' --nb_epoch '200' --object_scale '5.0' --no_object_scale '1.0' --coord_scale '1.0' --class_scale '1.0' --debug --log_path 'logs/' -- 'config_cryolo.json' '64'",
        f"{cmd_ini} train -c 'config_cryolo.json' -w '5' -g 1 -nc '4' --gpu_fraction '1.0' -e '10' -lft '2' --seed '10'",
        f"{cmd_ini} predict -c 'config_cryolo.json' -w 'cryolo_model_fromRelion_expand10.h5' -i tomograms -o '{output_dir}' -t '{thre}' -g '1' -d '0' -pbs '3' --gpu_fraction '1.0' -nc '4' --norm_margin '0.0' -sm 'LINE_STRAIGHTNESS' -st '0.95' -sr '1.41' -ad '10' --directional_method 'PREDICTED' -mw '100' --tomogram -tsr '-1' -tmem '0' -mn3d '2' -tmin '{connect_min}' -twin '-1' -tedge '0.4' -tmerge '0.8'"
    ]
    for cmd in commands:
        step_name = cmd.split()[2] 
        logging.info(f"--- Starting CryoLo step: {step_name} ---")
        run_command(cmd, log_file_path, cwd=cryolo_dir, module_load="cryolo")

    list_file = Path('ribo_list_final.txt')
    if not list_file.exists():
        logging.error(f"tomogram list file does not exist in the current directory: {list_file.resolve()}")
        sys.exit(1)
    
    with open(list_file, 'r') as f:
        for line in f.readlines():
            tomo, start_str, end_str, _ = line.strip().split()
            logging.info(f"Processing tomogram: {tomo}")

            coords_file = f"COORDS/{tomo}.coords"
            raw_star_file = f"STAR/{tomo}"
            (cryolo_dir / output_dir / "STAR" / tomo).mkdir(parents=True, exist_ok=True)

            cmd = [
                "cryolo_boxmanager_tools.py", "coords2star",
                "-i", coords_file, "-o", raw_star_file,
                "--apix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN)
            ]
            run_command(cmd, log_file_path, cwd=cryolo_dir / output_dir, module_load="cryolo")

            input_star_path = cryolo_dir / output_dir / raw_star_file / "particles_warp.star"
            filtered_star_file = f"STAR/{tomo}.star"
            output_star_path = cryolo_dir / output_dir / filtered_star_file
            
            logging.info(f"Filtering {input_star_path} to {output_star_path} with range {start_str}-{end_str}")

            try:
                start, end = float(start_str), float(end_str)
                with open(input_star_path, 'r') as infile, open(output_star_path, 'w') as outfile:
                    for star_line in infile:
                        parts = star_line.split()
                        if len(parts) < 4:
                            outfile.write(star_line)
                            continue
                        try:
                            z_coord = float(parts[3])
                            if start <= z_coord <= end:
                                outfile.write(star_line)
                        except ValueError:
                            outfile.write(star_line)
                logging.info(f"Successfully filtered {input_star_path}")
            except FileNotFoundError:
                logging.error(f"Could not find the star file to filter: {input_star_path}")
            except Exception as e:
                logging.error(f"An error occurred during star file filtering for {tomo}: {e}")

    logging.info("--- Starting WarpTools ts_export_particles ---")
    
    env = os.environ.copy()
    env['WARP_FORCE_MRC_FLOAT32'] = '1'
    
    angpix = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    
    cmd_export = [
        "WarpTools", "ts_export_particles",
        "--settings", "warp_tiltseries.settings",
        "--input_directory", f"{cryolo_dir}/{output_dir}/STAR/",
        "--input_processing", "warp_tiltseries",
        "--input_pattern", "*.star",
        "--coords_angpix", str(angpix),
        "--output_star", f"relion32_cryolo_expand/cryolo_{output_dir}.star",
        "--output_angpix", str(angpix),
        "--output_processing", "relion32_cryolo_expand/",
        "--box", "72",
        "--diameter", "350",
        "--relative_output_paths",
        "--device_list", str(cfg.gpu_devices[0]),
        "--perdevice", str(cfg.jobs_per_gpu),
        "--3d"
    ]
    run_command(cmd_export, log_file_path, env=env, module_load="warp/2.0.0dev31")
    logging.info("--- WarpTools ts_export_particles completed. ---")

def main():
    """Main function to initiate the appendix processing jobs."""
    parser = argparse.ArgumentParser(description="A stepwise handler for cryo-ET data processing.")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['isonet', 'cryolo', 'reconstruction'],
        help="Which stage of the pipeline to run."
    )
    args = parser.parse_args()

    if cfg.dataset_name != Path.cwd().name:
        logging.warning(
            f"'{cfg.dataset_name}' in config does not match '{Path.cwd().name}'."
        )

    logs_dir = Path("logs")
    if not logs_dir.exists():
        logging.warning(f"Logs directory {logs_dir} does not exist. Did you run the main pipeline?")

    try:
        log_file_path = logs_dir / f"{args.stage}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        logging.info(f"Main log file for this run is: {log_file_path.resolve()}")

        if args.stage == 'reconstruction':
            reconstruction(log_file_path)
        if args.stage == 'isonet':
            isonet(log_file_path)
        if args.stage == 'cryolo':
            cryolo(log_file_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
