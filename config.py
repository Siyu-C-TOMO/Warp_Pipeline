# ==================================================
# ============ USER-EDITABLE SETTINGS ==============
# ==================================================
# --- General Settings ---
dataset_name = "251028_HSC_2d"
raw_directory = "/data/Microscopy/Titan/Siyu" 
# Path to where you would like to save your raw data
# Titan2 data will be moved there and there might be no copy of your raw data in the original place
frame_folder = "frames"
mdoc_folder = "mdocs"
gain_ref = "wrong.gain"
tomo_match_string = "20251028_L" 

# --- Key Acquisition Parameters ---
angpix = 0.935
dose = 5.172
tilt_axis_angle = 84.48
thickness_pxl = 3000
camera_type = "Falcon4" # Switch between "K3" or "Falcon4"

# --- Falcon4 Specific Settings ---
# The source directory containing raw .eer and .eer.mdoc files
falcon4_source_dir = "/data/Microscopy/titan2/Villa_20251028_100000_SYC"
falcon4_eer_ngroups = 8

# --- K3 Specific Settings ---
k3_frame_num = 8

# ==================================================
# ============== ADVANCED PARAMETERS  ==============
# ==================================================

# --- Computing Resources ---
import os
gpu_devices = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
jobs_per_gpu = 1
etomo_cpu_cores = 8

# --- Warp Pipeline Parameters ---
pipeline_params = {}
extra_create_args = []

if camera_type == "K3":
    pipeline_params["extension"] = "*.tif"
    pipeline_params["m_grid_frames"] = k3_frame_num
    pipeline_params["original_x_y_size"] = (5760, 4092)
    extra_create_args.append("--gain_flip_x")

elif camera_type == "Falcon4":
    pipeline_params["extension"] = "*.eer"
    pipeline_params["m_grid_frames"] = falcon4_eer_ngroups
    pipeline_params["original_x_y_size"] = (4096, 4096)
    extra_create_args.extend(["--eer_ngroups", str(falcon4_eer_ngroups)])

else:
    raise ValueError(f"Unsupported camera_type in config: {camera_type}")

pipeline_params["extra_create_args"] = extra_create_args

# --- eTomo Settings ---
FINAL_NEWSTACK_BIN = 8
final_x_size = pipeline_params["original_x_y_size"][0] // FINAL_NEWSTACK_BIN
final_y_size = pipeline_params["original_x_y_size"][1] // FINAL_NEWSTACK_BIN

etomo_params = {
    "setupset.copyarg.pixel": angpix / 10.0,
    "setupset.copyarg.rotation": tilt_axis_angle,
    # "comparam.newst.newstack.SizeToOutputInXandY": f"{final_x_size},{final_y_size}",
}

use_dynamic_patch_size = True
patch_size_division_factor = 4
possible_patch_sizes = [256, 512, 1024, 2048]
default_patch_size = 512

# --- ISONet Settings ---
isonet_params = {
    "cube_size": 64,
    "crop_size": 96,
    "number_subtomos": 8,
    "iterations": 30,
    "noise_level": [0.1, 0.15, 0.2, 0.25],
    "noise_start_iter": [10, 15, 20, 25],
    "density_percentage": 40,
    "std_percentage": 40,
    "z_crop": 0.14,
    "batch_size": len(gpu_devices)
}

# --- 3D template matching Settings ---
template_matching_params = {
    "tomo_angpix": angpix * FINAL_NEWSTACK_BIN,
    "subdivisions": 3,
    "template_path": "/data/workspace/Siyu/Titan1_Processing/test/20250820_HSC_4hr_CorrectHand/relion32_linux/Refine3D/fc_clean_mr1/run_class001.mrc",
    "template_diameter": 350,
    "peak_distance": 175,
    "symmetry": "C1",
    "input_data": "NONE",  # Path to the matching.txt file; if "NONE", will run with full tomoset
}