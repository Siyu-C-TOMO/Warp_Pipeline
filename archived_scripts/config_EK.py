# ==================================================
# ============ USER-EDITABLE SETTINGS ==============
# ==================================================
# --- General Settings ---
dataset_name = "20240328_A549TNT_EK"
raw_directory = "/data/Microscopy/Titan" 
# Path to where you would like to save your raw data
# Titan2 data will be moved there and there might be no copy of your raw data in the original place
frame_folder = "frames"
mdoc_folder = "pacetomo"
gain_ref = "CountRef_pol1_g1_ts_001_001_000_5.0.mrc"
tomo_match_string = "pol10_2" 

# --- Key Acquisition Parameters ---
angpix = 1.635
dose = 5
tilt_axis_angle = -94.6
thickness_pxl = 3000
camera_type = "K3" # Switch between "K3" or "Falcon4"

# --- Falcon4 Specific Settings ---
# The source directory containing raw .eer and .eer.mdoc files
falcon4_source_dir = "/data/Microscopy/titan2/Villa_20251001_100000_SYC"
falcon4_eer_ngroups = 16

# --- K3 Specific Settings ---
k3_frame_num = 8

# --- Computing Resources ---
gpu_devices = [2]
jobs_per_gpu = 4
etomo_cpu_cores = 8

# ==================================================
# ========= DERIVED PARAMETERS (DO NOT EDIT) =========
# ==================================================

pipeline_params = {}
extra_create_args = []

if camera_type == "K3":
    pipeline_params["extension"] = "*.tif"
    pipeline_params["m_grid_frames"] = k3_frame_num
    pipeline_params["original_x_y_size"] = (5760, 4092)
    extra_create_args.append("--gain_flip_y")

elif camera_type == "Falcon4":
    pipeline_params["extension"] = "*.eer"
    pipeline_params["m_grid_frames"] = falcon4_eer_ngroups
    pipeline_params["original_x_y_size"] = (4096, 4096)
    extra_create_args.extend(["--eer_ngroups", str(falcon4_eer_ngroups)])

else:
    raise ValueError(f"Unsupported camera_type in config: {camera_type}")

pipeline_params["extra_create_args"] = extra_create_args

# --- eTomo Binning Calculation ---
FINAL_NEWSTACK_BIN = 8
final_x_size = pipeline_params["original_x_y_size"][0] // FINAL_NEWSTACK_BIN
final_y_size = pipeline_params["original_x_y_size"][1] // FINAL_NEWSTACK_BIN

etomo_params = {
    "setupset.copyarg.pixel": angpix / 10.0,
    "setupset.copyarg.rotation": tilt_axis_angle,
    # "comparam.newst.newstack.SizeToOutputInXandY": f"{final_x_size},{final_y_size}",
}

# --- eTomo Patch Size Settings ---
use_dynamic_patch_size = True
patch_size_division_factor = 8
possible_patch_sizes = [256, 512, 1024, 2048]
default_patch_size = 512