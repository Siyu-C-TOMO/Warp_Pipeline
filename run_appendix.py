#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from pathlib import Path

import config as cfg
from pipeline_utils import run_command, filter_star_file
from command_builder import (
    build_reconstruction_command,
    build_isonet_commands,
    build_cryolo_commands,
    build_particle_export_command,
    build_template_match_command,
)


def reconstruction(log_file_path: Path):
    """Runs the final reconstruction and packaging stage for Windows compatibility."""
    logging.info("Starting final reconstruction and packaging stage...")

    win_dir = Path("forWindows_frames")
    win_dir.mkdir(exist_ok=True)
    logging.info(f"{win_dir.resolve()} is ready for packaging.")

    logging.info("Running WarpTools ts_reconstruct...")
    env = os.environ.copy()
    
    cmd_reconstruct = build_reconstruction_command(cfg.jobs_per_gpu, cfg.gpu_devices)
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
    list_file = Path('ribo_list_final.txt')
    if not list_file.exists():
        logging.error(f"tomogram list file does not exist in the current directory: {list_file.resolve()}")
        sys.exit(1)
    
    tomo_list = [line.split()[0] for line in list_file.read_text().strip().splitlines()]
    logging.info(f"Found {len(tomo_list)} tomograms in the list.")

    tomo_folder = "tomoset"
    isonet_dir = Path("isonet")
    cmd_log_dir = isonet_dir / "logs"
    Path(cmd_log_dir).mkdir(parents=True, exist_ok=True)
    Path(isonet_dir / tomo_folder).mkdir(parents=True, exist_ok=True)

    for tomo in tomo_list:
        source = Path(f"warp_tiltseries/tiltstack/{tomo}").resolve()
        target = isonet_dir / tomo_folder / f"{tomo}.mrc"
        source_files = list(source.glob("*dev.mrc"))
        if not source_files:
            logging.warning(f"No _dev.mrc file found for tomogram {tomo}")
            continue
        if not target.exists():
            target.symlink_to(source_files[0])

    commands = build_isonet_commands(cfg.isonet_params, cfg.gpu_devices, tomo_folder)
    
    total_steps = len(commands)
    for i, cmd in enumerate(commands, 1):
        logging.info(f"--- Starting ISONet step [{i}/{total_steps}]: {cmd.split()[1]} ---")
        run_command(cmd, cmd_log_dir / f"step_{i}.log", cwd=isonet_dir, module_load='isonet')
    
    logging.info("--- All ISONet steps completed successfully. ---")

def cryolo(log_file_path: Path):
    """Run the Cryolo stage of the pipeline."""
    cryolo_dir = Path("cryolo")
    if not cryolo_dir.exists():
        logging.error(f"No cryolo directory. Run star-handler process-relion2cryolo first.")
        sys.exit(1)

    list_file = Path('ribo_list_final.txt')
    if not list_file.exists():
        logging.error(f"tomogram list file does not exist in the current directory: {list_file.resolve()}")
        sys.exit(1)

    commands, output_dir = build_cryolo_commands(cfg.cryolo_params, cfg.gpu_devices)
    
    cmd_log_dir = cryolo_dir / "logs"
    Path(cmd_log_dir).mkdir(parents=True, exist_ok=True)
    total_steps = len(commands)
    for i, cmd in enumerate(commands, 1):
        step_name = cmd.split()[4] 
        logging.info(f"--- Starting CryoLo step [{i}/{total_steps}]: {step_name} ---")
        # run_command(cmd, cmd_log_dir / f"step_{i}.log", cwd=cryolo_dir, module_load="cryolo")
    
    with open(list_file, 'r') as f:
        to_star_log_dir = cmd_log_dir / "to_star"
        Path(to_star_log_dir).mkdir(parents=True, exist_ok=True)

        for line in f.readlines():
            tomo, start_str, end_str, _ = line.strip().split()
            logging.info(f"Processing tomogram: {tomo}")

            coords_file = f"COORDS/{tomo}.coords"
            raw_star_dir = cryolo_dir / output_dir / "STAR" / tomo
            raw_star_dir.mkdir(parents=True, exist_ok=True)
            raw_star_file = raw_star_dir / "particles_warp.star"

            cmd_coords = [
                "cryolo_boxmanager_tools.py", "coords2star",
                "-i", str(cryolo_dir / output_dir / coords_file),
                "-o", str(raw_star_dir),
                "--apix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN)
            ]
            run_command(cmd_coords, to_star_log_dir / f"{tomo}.log", module_load="cryolo")

            filtered_star_file = cryolo_dir / output_dir / "STAR" / f"{tomo}.star"
            logging.info(f"Filtering {raw_star_file} to {filtered_star_file} with range {start_str}-{end_str}")
            
            try:
                z_range = (float(start_str), float(end_str))
                filter_star_file(raw_star_file, filtered_star_file, z_range)
            except ValueError:
                logging.error(f"Invalid range for tomogram {tomo}: {start_str}-{end_str}")

    logging.info("--- Starting WarpTools ts_export_particles ---")
    env = os.environ.copy()
    env['WARP_FORCE_MRC_FLOAT32'] = '1'
    
    cmd_export = build_particle_export_command(cryolo_dir, output_dir, cfg.jobs_per_gpu, cfg.gpu_devices)
    run_command(cmd_export, cmd_log_dir / "export.log", env=env, module_load="warp/2.0.0dev36")
    logging.info("--- WarpTools ts_export_particles completed. ---")

def template_match_3D(log_file_path: Path):
    """Run the 3D template matching stage of the pipeline."""
    template_path = Path(cfg.template_matching_params['template_path'])
    if not template_path.exists():
        logging.error(f"Template file does not exist: {template_path.resolve()}")
        sys.exit(1)

    list_file = Path(cfg.template_matching_params['input_data'])
    if not list_file.exists():
        logging.warning(f"no list file available: {list_file.resolve()}. Running with full tomoset.")

    logging.info("Running WarpTools ts_template_match...")
    env = os.environ.copy()
    cmd_template_match = build_template_match_command(
        cfg.template_matching_params, cfg.jobs_per_gpu, cfg.gpu_devices
    )
    
    logging.info(f"--- Starting Warp 3D template matching ---")
    run_command(cmd_template_match, log_file_path, env=env, module_load="warp/2.0.0dev36")
    logging.info("--- WarpTools ts_template_match completed. ---")

def main():
    """Main function to initiate the appendix processing jobs."""
    parser = argparse.ArgumentParser(description="A stepwise handler for cryo-ET data processing.")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['isonet', 'cryolo', 'reconstruction', '3DTM'],
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

        stage_map = {
            'reconstruction': reconstruction,
            'isonet': isonet,
            'cryolo': cryolo,
            '3DTM': template_match_3D,
        }
        if args.stage in stage_map:
            stage_map[args.stage](log_file_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
