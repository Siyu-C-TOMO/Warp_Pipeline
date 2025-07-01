#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import subprocess
import glob
import shutil
import config as cfg

def run_preprocess():
    """Runs the preprocessing stage, assuming CWD is the dataset directory."""
    logging.info("Starting preprocessing stage...")
    
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.info("Linking data files...")
    os.makedirs("mdocs", exist_ok=True)
    os.makedirs("frames", exist_ok=True)
    
    mdoc_source_path = os.path.join(cfg.raw_directory, cfg.dataset_name, cfg.mdoc_folder)
    for mdoc_file in glob.glob(os.path.join(mdoc_source_path, f"{cfg.tomo_match_string}*ts_???.mrc.mdoc")):
        base_name = os.path.basename(mdoc_file).replace(".mrc.", ".")
        if not os.path.lexists(os.path.join("mdocs", base_name)):
            os.symlink(mdoc_file, os.path.join("mdocs", base_name))

    frame_source_path = os.path.join(cfg.raw_directory, cfg.dataset_name, cfg.frame_folder)
    for frame_file in glob.glob(os.path.join(frame_source_path, f"{cfg.tomo_match_string}*")):
    #for frame_file in glob.glob(os.path.join(frame_source_path, "L1_G1_ts_00*")):
        if not os.path.lexists(os.path.join("frames", os.path.basename(frame_file))):
            os.symlink(frame_file, os.path.join("frames", os.path.basename(frame_file)))
    
    gain_ref_path = os.path.join(frame_source_path, cfg.gain_ref)
    if not os.path.lexists(os.path.join("frames", cfg.gain_ref)):
        os.symlink(gain_ref_path, os.path.join("frames", cfg.gain_ref))

    logging.info("Creating frame series settings...")
    cmd_frame_settings = [
        "WarpTools", "create_settings",
        "--folder_data", "frames",
        "--folder_processing", "warp_frameseries",
        "--output", "warp_frameseries.settings",
        "--extension", "*.eer",
        "--angpix", str(cfg.angpix),
        "--exposure", str(cfg.dose),
        "--gain_path", os.path.join("frames", cfg.gain_ref),
        "--eer_ngroups", str(cfg.eer_ngroups)
    ]
    with open("logs/frame_settings.log", "w") as log_file:
        subprocess.run(cmd_frame_settings, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Creating tilt series settings...")
    os.makedirs("tomostar", exist_ok=True)
    cmd_tilt_settings = [
        "WarpTools", "create_settings",
        "--output", "warp_tiltseries.settings",
        "--folder_processing", "warp_tiltseries",
        "--folder_data", "tomostar",
        "--extension", "*.tomostar",
        "--angpix", str(cfg.angpix),
        "--gain_path", os.path.join("frames", cfg.gain_ref),
        "--exposure", str(cfg.dose),
        "--tomo_dimensions", f"{cfg.original_x_y_size[0]}x{cfg.original_x_y_size[1]}x{cfg.thickness_pxl}"
    ]
    with open("logs/tilt_settings.log", "w") as log_file:
        subprocess.run(cmd_tilt_settings, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Running frame series motion and CTF estimation...")
    cmd_motion_ctf = [
        "WarpTools", "fs_motion_and_ctf",
        "--settings", "warp_frameseries.settings",
        "--m_grid", f"1x1x{cfg.eer_ngroups}",
        "--c_grid", "2x2x1",
        "--c_range_max", "7",
        "--c_defocus_max", "8",
        "--c_use_sum",
        "--out_averages",
        "--out_average_halves",
        "--device_list", str(cfg.gpu_devices[0]),
        "--perdevice", str(cfg.jobs_per_gpu*2)  # Adjusted for motion and CTF
    ]
    with open("logs/motion_ctf.log", "w") as log_file:
        subprocess.run(cmd_motion_ctf, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Plotting histograms of 2D processing metrics...")
    cmd_histograms = [
        "WarpTools", "filter_quality",
        "--settings", "warp_frameseries.settings",
        "--histograms"
    ]
    with open("logs/histograms.log", "w") as log_file:
        subprocess.run(cmd_histograms, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Importing tilt series metadata...")
    cmd_ts_import = [
        "WarpTools", "ts_import",
        "--mdocs", "mdocs",
        "--frameseries", "warp_frameseries",
        "--tilt_exposure", str(cfg.dose),
        "--dont_invert",
        "--output", "tomostar"
    ]
    with open("logs/tomostar.log", "w") as log_file:
        subprocess.run(cmd_ts_import, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Preprocessing stage completed.")

def run_builtin_etomo():
    """Runs the eTomo alignment stage with a patched environment for testing."""
    logging.info("Starting Patched eTomo alignment stage for testing...")

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_dir = os.path.join(pipeline_dir, 'imod_wrappers')
    if not os.path.isdir(wrapper_dir):
        logging.error(f"IMOD wrapper directory not found at: {wrapper_dir}")
        return

    env = os.environ.copy()
    env['PATH'] = f"{wrapper_dir}{os.pathsep}{env.get('PATH', '')}"   

    cmd_to_run = [
        "WarpTools", "ts_etomo_patches",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix*cfg.FINAL_NEWSTACK_BIN),
        "--initial_axis", str(cfg.tilt_axis_angle), 
        "--patch_size", "512",
#        "--do_axis_search",
        "--device_list", str(cfg.gpu_devices[0]),
        "--perdevice", str(cfg.jobs_per_gpu*2)  # Adjusted for eTomo patches
    ]
    
    with open("logs/etomo_patches.log", "w") as log_file:
        subprocess.run(cmd_to_run, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    logging.info("Patched eTomo test stage completed.")

def optimize_etomo():
    """Runs the customized eTomo optimization stage."""
    logging.info("Starting eTomo optimization stage...")
    
    original_dir = os.getcwd()
    etomo_dir = os.path.join(original_dir, "warp_tiltseries", "tiltstack")
    
    if not os.path.isdir(etomo_dir):
        logging.error(f"eTomo directory not found at: {etomo_dir}")
        logging.error("Please ensure the preprocessing stage was run successfully.")
        return

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etomo_optimize.py')
    log_path = os.path.join(original_dir, 'logs', 'etomo_optimization.log')

    try:
        logging.info(f"Changing directory to {etomo_dir} to run optimization script.")
        os.chdir(etomo_dir)
        
        with open(log_path, 'w') as log_file:
            subprocess.run(
                [sys.executable, script_path],
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            
    except subprocess.CalledProcessError as e:
        logging.error(f"eTomo optimization script failed with exit code {e.returncode}.")
        logging.error(f"Check the log for details: {log_path}")
    except FileNotFoundError:
        logging.error(f"Could not find the optimization script at {script_path}")
    finally:
        logging.info(f"Returning to directory: {original_dir}")
        os.chdir(original_dir)
        
    logging.info("eTomo optimization stage completed.")

def run_postprocess():
    """Runs the post-processing stage, assuming CWD is the dataset directory."""
    logging.info("Starting post-processing stage...")
    
    logging.info("Extracting RMD and saving table...")
    rmd_error_log = "logs/eTomo_RMD_error.txt"
    if os.path.exists(rmd_error_log):
        os.rename(rmd_error_log, rmd_error_log + "~")
    
    with open(rmd_error_log, "w") as f:
        for d in glob.glob("warp_tiltseries/tiltstack/L*"):
            align_log = os.path.join(d, "align_clean.log")
            if os.path.exists(align_log):
                with open(align_log, "r") as log_file:
                    for line in log_file:
                        if "Residual error weighted mean" in line:
                            new_rm = line.split()[-2]
                            f.write(f"{os.path.basename(d)}\t{new_rm}\n")
                            break

    logging.info("Importing improved alignments...")
    cmd_import_align = [
        "WarpTools", "ts_import_alignments",
        "--settings", "warp_tiltseries.settings",
        "--alignments", "warp_tiltseries/tiltstack/",
        "--alignment_angpix", str(cfg.angpix)
    ]
    with open("logs/import_align.log", "w") as log_file:
        subprocess.run(cmd_import_align, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Checking defocus handedness...")
    cmd_hand_check = [
        "WarpTools", "ts_defocus_hand",
        "--settings", "warp_tiltseries.settings",
        "--check"
    ]
    result = subprocess.run(cmd_hand_check, check=True, capture_output=True, text=True)
    with open("logs/handness.log", "w") as log_file:
        log_file.write(result.stdout)
    
    handness = result.stdout.strip().split("'")[-2]
    if handness != "no flip":
        logging.info("Flipping tomograms...")
        cmd_hand_flip = [
            "WarpTools", "ts_defocus_hand",
            "--settings", "warp_tiltseries.settings",
            "--set_flip"
        ]
        with open("logs/flipped.log", "w") as log_file:
            subprocess.run(cmd_hand_flip, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        logging.info("No need to flip tomograms.")

    logging.info("Estimating tilt series CTF...")
    cmd_ts_ctf = [
        "WarpTools", "ts_ctf",
        "--settings", "warp_tiltseries.settings",
        "--range_high", "7",
        "--defocus_max", "8",
        "--device_list", str(cfg.gpu_devices[0]),
        "--perdevice", str(cfg.jobs_per_gpu)
    ]
    with open("logs/tomo_ctf.log", "w") as log_file:
        subprocess.run(cmd_ts_ctf, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    logging.info("Parsing XML files to remove bad tilts...")
    xml_backup_dir = "warp_tiltseries/tiltstack/xml_backup"
    if not os.path.exists(xml_backup_dir):
        os.makedirs(xml_backup_dir)
        for xml_file in glob.glob("warp_tiltseries/*.xml"):
            shutil.copy(xml_file, os.path.join(xml_backup_dir, os.path.basename(xml_file)))
    else:
        for xml_file in glob.glob(os.path.join(xml_backup_dir, "*.xml")):
            shutil.copy(xml_file, os.path.join("warp_tiltseries", os.path.basename(xml_file)))
    
    original_dir = os.getcwd()
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xml_parser.py')
    log_path = os.path.join(os.getcwd(), 'logs', 'xml_parsing.log')
    
    original_dir = os.getcwd()
    xml_dir = os.path.join(original_dir, "warp_tiltseries", "tiltstack")

    try:
        logging.info(f"Changing directory to {xml_dir} to run XML parsing script.")
        os.chdir(xml_dir)
        
        with open(log_path, 'w') as log_file:
            subprocess.run(
                [sys.executable, script_path], 
                check=True, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )

    except subprocess.CalledProcessError as e:
        logging.error(f"XML parsing script failed with exit code {e.returncode}.")
        logging.error(f"Check the log for details: {log_path}")
    except FileNotFoundError:
        logging.error(f"Could not find the XML parsing script at {script_path}")
    finally:
        logging.info(f"Returning to directory: {original_dir}")
        os.chdir(original_dir)

    logging.info("Applying deconvolution...")
    with open("logs/deconv.log", "w") as log_file:
        for xml_file in glob.glob("warp_tiltseries/*.xml"):
            tomo_name = os.path.basename(xml_file).split('.')[0]
            defocus = ""
            with open(xml_file, "r") as f:
                for line in f:
                    if 'Defocus" Val' in line:
                        defocus = line.split('"')[3]
                        break
            
            if not defocus:
                logging.warning(f"Defocus not found for {tomo_name}, skipping deconvolution.")
                continue
            
            tomo_dir = f"warp_tiltseries/tiltstack/{tomo_name}"
            rec_file = f"{tomo_name}_rot_flipz.mrc"
            mrc_file = f"{tomo_name}_rot_flipz_dev.mrc"

            rec_file_full_path = os.path.join(tomo_dir, rec_file)
            if not os.path.exists(rec_file_full_path):
                logging.warning(f"Input file not found, skipping deconvolution for {tomo_name}: {rec_file_full_path}")
                continue
            
            command_string = (
                "module unload imod; "
                "module load imod/5.0.1-beta; "
                "reducefiltvol "
                f"-i {rec_file} "
                f"-o {mrc_file} "
                "-dec 0.5 "
                f"-def {defocus}"
            )
            
            try:
                os.chdir(tomo_dir)
                subprocess.run(command_string, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logging.error(f"Deconvolution command failed for {tomo_name} with exit code {e.returncode}. See logs/deconv.log for details.")
            finally:
                os.chdir(original_dir)

            for f in glob.glob(os.path.join(tomo_dir, "*~")):
                os.remove(f)

    logging.info("Post-processing stage completed.")

def main():
    """Main function to drive the pipeline."""
    parser = argparse.ArgumentParser(description="A flexible pipeline for cryo-ET data processing.")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['preprocess', 'etomo', 'optimize', 'postprocess', 'all'],
        default='all',
        help="Which stage of the pipeline to run."
    )
    args = parser.parse_args()

    original_dir = os.getcwd()
    dataset_dir = os.path.join(original_dir, cfg.dataset_name)

    try:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        os.chdir(dataset_dir)

        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, "pipeline.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Changing working directory to {dataset_dir}")
        logging.info(f"Main log file for this run is: {os.path.abspath(log_file_path)}")

        if args.stage in ['all', 'preprocess']:
            run_preprocess()
        if args.stage in ['all', 'etomo']:
            run_builtin_etomo()
        if args.stage in ['all', 'optimize']:
            optimize_etomo()
        if args.stage in ['all', 'postprocess']:
            run_postprocess()

    finally:
        logging.info(f"Returning to original directory: {original_dir}")
        os.chdir(original_dir)
        logging.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
