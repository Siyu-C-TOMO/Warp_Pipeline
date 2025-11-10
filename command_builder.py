#!/usr/bin/env python3

from pathlib import Path
import config as cfg

def build_reconstruction_command(jobs_per_gpu, gpu_devices):
    """Builds the command for the reconstruction stage."""
    cmd = [
        "WarpTools", "ts_reconstruct",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_isonet_commands(isonet_params, gpu_devices, tomo_folder="tomoset"):
    """Builds the list of commands for the ISONet stage."""
    pixel_size = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    gpu_ids = ','.join(str(d) for d in gpu_devices)
    noise_level = ','.join(str(x) for x in isonet_params['noise_level'])
    noise_start_iter = ','.join(str(x) for x in isonet_params['noise_start_iter'])
    
    commands = [
        f"isonet.py prepare_star {tomo_folder} --output_star tomogram.star --pixel_size {pixel_size} --number_subtomos {isonet_params['number_subtomos']}",
        f"isonet.py make_mask tomogram.star --mask_folder mask --density_percentage {isonet_params['density_percentage']} --std_percentage {isonet_params['std_percentage']} --z_crop {isonet_params['z_crop']}",
        f"isonet.py extract tomogram.star --cube_size {isonet_params['cube_size']}",
        f"isonet.py refine subtomo.star --gpuID {gpu_ids} --iterations {isonet_params['iterations']} --noise_level {noise_level} --noise_start_iter {noise_start_iter} --log_level info --batch_size {isonet_params['batch_size']}",
        f"isonet.py predict tomogram.star ./results/model_iter{isonet_params['iterations']}.h5 --gpuID {gpu_ids} --cube_size {isonet_params['cube_size']} --crop_size {isonet_params['crop_size']} --log_level info --batch_size {isonet_params['batch_size']}",
    ]
    return commands

def build_cryolo_commands(cryolo_params):
    """Builds the list of commands for the Cryolo stage."""
    threshold = cryolo_params['threshold']
    min_connections = cryolo_params['min_connections']
    
    cryolo_ad = "/software/repo/rhel9/cryolo/1.9.4/bin"
    cmd_ini = f"'{cryolo_ad}/python3.8' -u '{cryolo_ad}/cryolo_gui.py' --ignore-gooey"
    output_dir = Path(f"expand10_{threshold}_{min_connections}")

    commands = [
        f"{cmd_ini} config --train_image_folder '3DTM_pre/tomograms' --train_annot_folder '3DTM_pre/CBOX' --saved_weights_name 'cryolo_model_fromRelion_expand10.h5' -a 'PhosaurusNet' --input_size 1024 -nm 'STANDARD' --num_patches '1' --overlap_patches '200' --filtered_output 'filtered_tmp/' -f 'LOWPASS' --low_pass_cutoff '0.1' --janni_overlap '24' --janni_batches '3' --train_times '10' --batch_size '4' --learning_rate '0.0001' --nb_epoch '200' --object_scale '5.0' --no_object_scale '1.0' --coord_scale '1.0' --class_scale '1.0' --debug --log_path 'logs/' -- 'config_cryolo.json' '64'",
        f"{cmd_ini} train -c 'config_cryolo.json' -w '5' -g 1 -nc '4' --gpu_fraction '1.0' -e '10' -lft '2' --seed '10'",
        f"{cmd_ini} predict -c 'config_cryolo.json' -w 'cryolo_model_fromRelion_expand10.h5' -i tomograms -o '{output_dir}' -t '{threshold}' -g '1' -d '0' -pbs '3' --gpu_fraction '1.0' -nc '4' --norm_margin '0.0' -sm 'LINE_STRAIGHTNESS' -st '0.95' -sr '1.41' -ad '10' --directional_method 'PREDICTED' -mw '100' --tomogram -tsr '-1' -tmem '0' -mn3d '2' -tmin '{min_connections}' -twin '-1' -tedge '0.4' -tmerge '0.8'"
    ]
    return commands, output_dir

def build_particle_export_command(cryolo_dir, output_dir, jobs_per_gpu, gpu_devices):
    """Builds the command for exporting particles with WarpTools."""
    angpix = cfg.angpix * cfg.FINAL_NEWSTACK_BIN
    cmd = [
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
        "--perdevice", str(jobs_per_gpu),
        "--3d"
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_template_match_command(params, jobs_per_gpu, gpu_devices):
    """Builds the command for the 3D template matching stage."""
    cmd = [
        "WarpTools", "ts_template_match",
        "--settings", "warp_tiltseries.settings",
        "--tomo_angpix", str(params['tomo_angpix']),
        "--subdivisions", str(params['subdivisions']),
        "--template_path", str(params['template_path']),
        "--template_diameter", str(params['template_diameter']),
        "--peak_distance", str(params['peak_distance']),
        "--symmetry", str(params['symmetry']),
        "--perdevice", str(jobs_per_gpu),
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    
    list_file = Path(params['input_data'])
    if list_file.exists():
        cmd.extend(["--input_data", str(list_file)])
    
    return cmd

# --- Commands for run_pipeline.py ---

def build_frame_settings_command(frame_source_path, params):
    """Builds the command to create frame series settings."""
    cmd = [
        "WarpTools", "create_settings",
        "--folder_data", frame_source_path,
        "--folder_processing", "warp_frameseries",
        "--output", "warp_frameseries.settings",
        "--extension", params["extension"],
        "--angpix", str(cfg.angpix),
        "--exposure", str(cfg.dose),
    ]
    cmd.extend(params["extra_create_args"])
    return cmd

def build_tilt_settings_command(params):
    """Builds the command to create tilt series settings."""
    return [
        "WarpTools", "create_settings",
        "--output", "warp_tiltseries.settings",
        "--folder_processing", "warp_tiltseries",
        "--folder_data", "tomostar",
        "--extension", "*.tomostar",
        "--angpix", str(cfg.angpix),
        "--exposure", str(cfg.dose),
        "--tomo_dimensions", f"{params['original_x_y_size'][0]}x{params['original_x_y_size'][1]}x{cfg.thickness_pxl}"
    ]

def build_motion_ctf_command(params, jobs_per_gpu, gpu_devices):
    """Builds the command for motion correction and CTF estimation."""
    cmd = [
        "WarpTools", "fs_motion_and_ctf",
        "--settings", "warp_frameseries.settings",
        "--m_grid", f"1x1x{params['m_grid_frames']}",
        "--c_grid", "2x2x1",
        "--c_range_max", "7",
        "--c_defocus_max", "8",
        "--c_use_sum",
        "--out_averages",
        "--out_average_halves",
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_histograms_command():
    """Builds the command to plot histograms."""
    return [
        "WarpTools", "filter_quality",
        "--settings", "warp_frameseries.settings",
        "--histograms"
    ]

def build_ts_import_command():
    """Builds the command to import tilt series metadata."""
    return [
        "WarpTools", "ts_import",
        "--mdocs", "mdocs",
        "--frameseries", "warp_frameseries",
        "--tilt_exposure", str(cfg.dose),
        "--dont_invert",
        "--override_axis", str(cfg.tilt_axis_angle),
        "--output", "tomostar"
    ]

def build_ts_etomo_patches_command(patch_size, jobs_per_gpu, gpu_devices):
    """Builds the command for eTomo patch-based alignment."""
    cmd = [
        "WarpTools", "ts_etomo_patches",
        "--settings", "warp_tiltseries.settings",
        "--angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
        "--initial_axis", str(cfg.tilt_axis_angle),
        "--patch_size", str(patch_size),
        "--perdevice", str(jobs_per_gpu * 2)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd

def build_ts_stack_command():
    """Builds the command to create tilt stacks."""
    return [
        "WarpTools", "ts_stack",
        "--settings", "warp_tiltseries.settings"
    ]

def build_import_align_command():
    """Builds the command to import improved alignments."""
    return [
        "WarpTools", "ts_import_alignments",
        "--settings", "warp_tiltseries.settings",
        "--alignments", "warp_tiltseries/tiltstack/",
        "--alignment_angpix", str(cfg.angpix * cfg.FINAL_NEWSTACK_BIN),
    ]

def build_hand_check_command():
    """Builds the command to check defocus handedness."""
    return [
        "WarpTools", "ts_defocus_hand",
        "--settings", "warp_tiltseries.settings",
        "--check"
    ]

def build_hand_flip_command():
    """Builds the command to flip tomograms if handedness is wrong."""
    return [
        "WarpTools", "ts_defocus_hand",
        "--settings", "warp_tiltseries.settings",
        "--set_flip"
    ]

def build_ts_ctf_command(jobs_per_gpu, gpu_devices):
    """Builds the command to estimate tilt series CTF."""
    cmd = [
        "WarpTools", "ts_ctf",
        "--settings", "warp_tiltseries.settings",
        "--range_high", "7",
        "--defocus_max", "8",
        "--perdevice", str(jobs_per_gpu)
    ]
    cmd.extend(["--device_list"] + [str(d) for d in gpu_devices])
    return cmd
