# ==================================================
# ============ MAIN CONFIGURATION ==================
# ==================================================
# --- General Settings ---
dataset_name = "250624_Proj_titan2"
raw_directory = "/data/Microscopy/Titan/Siyu"
frame_folder = "frames"
mdoc_folder = "mdocs"
gain_ref = "20250602_143520_EER_GainReference.gain"
tomo_match_string = "L"

# --- Key Acquisition Parameters ---
# These are the most common parameters you will need to change.
angpix = 0.935
dose = 5.172
eer_ngroups = 8
tilt_axis_angle = -94.881
thickness_pxl = 3000
camera_type = "Falcon4" # "K3" or "Falcon4"

# --- Computing Resources ---
gpu_devices = [2]
jobs_per_gpu = 4
etomo_cpu_cores = 8

# ==================================================
# ====== ETOMO PARAMETER CALCULATION LOGIC =========
# ==================================================
# This section calculates the final eTomo parameters.
# You should not need to edit this part directly.
if camera_type == "K3":
    original_x_y_size = (5760, 4092)
elif camera_type == "Falcon4":
    original_x_y_size = (4096, 4096)
else:
    original_x_y_size = (4096, 4096) # Default fallback

FINAL_NEWSTACK_BIN = 8

final_x_size = original_x_y_size[0] // FINAL_NEWSTACK_BIN
final_y_size = original_x_y_size[1] // FINAL_NEWSTACK_BIN

etomo_params = {
    "setupset.copyarg.pixel": angpix / 10.0,
    "setupset.copyarg.rotation": tilt_axis_angle,    
    "comparam.newst.newstack.SizeToOutputInXandY": f"{final_x_size},{final_y_size}",
}
