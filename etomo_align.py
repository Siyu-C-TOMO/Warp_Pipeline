#!/usr/bin/env python

import glob
import os
import sys
sys.path.insert(0, os.getcwd())
import config as cfg
import subprocess
from default_adoc import PARAMS as default_adoc_params

####################################
####### FUNCTION BLOCK BELOW #######

def generate_adoc_file(output_path):
    """
    Generates a .adoc file by merging default parameters with calculated ones from config.
    """
    final_params = default_adoc_params.copy()
    final_params.update(cfg.etomo_params)

    with open(output_path, 'w') as f:
        for key, value in final_params.items():
            f.write(f"{key} = {value}\n")

def make_etomo_files() -> tuple[list[str], str]:
    """
    Generates .adoc files for each tomogram and creates the batch command file.
    """
    pattern = f'{cfg.tomo_match_string}*'
    tomo_list = sorted(glob.glob(pattern))
    if not tomo_list:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    batch_string = cfg.dataset_name
    p = os.getcwd()

    etomo_com = f'{batch_string}_batch.com'
    with open(etomo_com, 'w') as f:
        f.write('$batchruntomo -StandardInput\n')
        f.write('NamingStyle	0\n')
        f.write(f'CheckFile	{batch_string}.cmds\n')
        f.write(f'CPUMachineList	localhost:{cfg.etomo_cpu_cores}\n')
        f.write('NiceValue	15\n')
        f.write('EtomoDebug	0\n')
        f.write('EndingStep	6.0\n')
        for ts in tomo_list:
            adoc_path = os.path.join(ts, f"{batch_string}_{ts}.adoc")
            generate_adoc_file(adoc_path)
            f.write(f'DirectiveFile\t{p}/{adoc_path}\n')
            f.write(f'RootName\t{ts}\n')
            f.write(f'CurrentLocation\t{p}/{ts}\n')
            
    print(f"Successfully created {etomo_com} and generated .adoc files for {len(tomo_list)} tomograms.")
    return tomo_list, etomo_com

def submfg(file_name):
    """SUBMIT ETOMO RELATED JOBS WITH CONTROLLED LOG FILES"""
    print(f"Starting Command {file_name} using etomo")
    result = subprocess.run(['submfg', file_name], check=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Command {file_name} ran successfully in the script.")
    else:
        print(f"Warning! Command {file_name} didn't run successfully.")

def rotation_axis() -> float:
    """USE align.log TO FIND THE ROTATION AXIS VALUE"""
    with open('align.log', 'r') as file:
        data = file.readlines()
        for line in data:
            if line.lstrip().startswith('At minimum tilt, rotation angle is'):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        print("Error: Unable to convert rotation axis value to float.")
                        return 0.0

####### FUNCTION BLOCK ABOVE #######
####################################

def run_alignment():
    """Main function to run the eTomo alignment process."""
    tomo_list, etomo_com = make_etomo_files()

    print("Starting batchetomo process...")
    submfg(etomo_com)
    print("All tasks are done.")
