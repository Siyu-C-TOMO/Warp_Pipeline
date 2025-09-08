#!/usr/bin/env python
#
# Wrapper for batchtomo
# 
# Before proceeding:
# - have all TS in their own directory (Warp default file structure)
# - have a batch session established for one TS with desired settings, stopping at fine alignment - adoc will be used.
#
# This script will:
# - copy the adoc from the examplar TS to all TS directories
# - add all TS to batch com file
# - run batchruntomo (up to fine alignment)
# - high-residual and view removal
# - edit aligncoms to ignore tilt angle offset
# - re-run aligncom
# - newstack for aligned stack (bin8)
# - reconstruct with SIRTlike iterations (bin8)
#
# JH 20230920
# 
# Add a manually input string to define the tomogram list 
# SYC 241129
# Fix a bug when early optimization jobs throw errors and prevent the following jobs from working
# Support multiprocessing to speed up the optimization process
# SYC 241211
# Bug fixed when batchruntomo.log is not available
# SYC 241218
#
# Refactor Note (June 2025):
# This script has been integrated into the Warp_Pipeline_v2 framework.
# It no longer relies on external templates. Instead, it dynamically generates
# .adoc and .com file parameters based on settings in 'config.py' and
# default values in 'default_adoc.py', providing a more flexible and
# robust workflow.

import glob
import os
import time
import pandas as pd
import numpy as np
import config as cfg
import subprocess
from multiprocessing import Pool
import csv
from default_adoc import PARAMS as default_adoc_params

# thresholds for pruning views and contours
view_thr_sd = 2
contour_thr_sd = 2


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

def makealigncom(aligncom,aligncomout):
    """MAKE ALIGN COM - WITHOUT TILT OFFSET, WITH UPDATED ROTATION AXIS
    
    Find the 'AngleOffset' row and replace the value to 0
    """
    angle_set = 'AngleOffset\t0\n'
    new_rotation_angle = rotation_axis()
    rotation_angle_set = f'RotationAngle\t{new_rotation_angle}\n'
    global cc
    with open(aligncom, 'r') as file:
        data = file.readlines()
        c = 0
        for line in data:
            if line.startswith('AngleOffset'):
                cc = c
            if line.startswith('RotationAngle'):
                data[c] = rotation_angle_set if rotation_angle_set else line
            if line.startswith('RotOption'):
                data[c] = 'RotOption\t0\n'
            c+=1
    data[cc] = angle_set
    with open(aligncomout, 'w', encoding='utf-8') as file:
        file.writelines(data)


def makenewstcom(newstcom,newstcomout):
    """MAKE NEWSTACK COM
    
    Add bin-8 related rows (bin and final size) to the file, using values from config.
    """
    # Dynamically create the strings using values from config.py
    size_string = f"SizeToOutputInXandY\t{cfg.final_x_size},{cfg.final_y_size}\n"
    bin_string = f"BinByFactor\t{cfg.FINAL_NEWSTACK_BIN}\n"
    
    global cc
    with open(newstcom, 'r') as file:
        data = file.readlines()
        c = 0
        for line in data:
            if line.startswith('SizeToOutputInXandY'):
                cc = c
                if not any("BinByFactor" in s for s in data):
                    data.insert(cc + 1, bin_string)
            c+=1
            
    data[cc] = size_string
    
    with open(newstcomout, 'w', encoding='utf-8') as file:
        file.writelines(data)

def maketiltcom(tiltcom,tiltcomout):
    """MAKE TILT COM FOR RECONSTRUCTION
    
    Add bin-8 related row to the file, using values from config.
    """
    angle_set = f'IMAGEBINNED\t{cfg.FINAL_NEWSTACK_BIN}\n'
    global cc
    with open(tiltcom, 'r') as file:
        data = file.readlines()
        c = 0
        for line in data:
            if line.startswith('IMAGEBINNED'):
                cc = c
            c+=1
    data[cc] = angle_set
    with open(tiltcomout, 'w', encoding='utf-8') as file:
        file.writelines(data)


def read_fiducial_file(path_to_fiducial_file):
    """READ FIDUCIAL FILE AS CSV FOR FURTHER PROCESSING
    """
    with open(path_to_fiducial_file, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    
    fidpts = []
    
    for i in range(0,len(data)):
        temp = [float(x) for x in data[i][0].split()]
        fidpts.append(temp)
    fidptsn = np.array(fidpts)
    fiducial_file_df = pd.DataFrame(fidptsn).astype('float64')
    
    return fiducial_file_df

 
def prepare_fiducial_file_to_write(fiducial_file_df,output_name):
    """PREPARE FIDUCIAL FILE TO WRITE"""
    fiducial_file_df[0] = fiducial_file_df[0].astype('int')
    fiducial_file_df[1] = fiducial_file_df[1].astype('int')
    fiducial_file_df.to_csv(output_name,index=False,header=False,sep ='\t') 


def read_align_log(path_to_align_log):
    """READ ALIGN LOG - SEPARATE DF FOR VIEWS AND CONTOURS"""
    with open(path_to_align_log,'r') as file:
        data = file.readlines()
    
    viewList = []
    for i in range(0,len(data)):
        lsplit = data[i].split()
        if len(lsplit) > 0:
            if lsplit[0] == "view":
                c = 1
                next_lsplit = data[i+c].split()
                while len(data[i+c].split())>0:
                    viewList.append(data[i+c].split())
                    c+=1       
        else:
            continue
    vn = np.array(viewList)
    view_df = pd.DataFrame(vn).astype('float64')
    
    cList = []
    for i in range(0,len(data)):
        lsplit = data[i].split()
        if len(lsplit) > 0:
            if lsplit[0] == "3-D":
                c = 2
                next_lsplit = data[i+c].split()
                while len(data[i+c].split())>0:
                    cList.append(data[i+c].split())
                    c+=1       
        else:
            continue
    cn = np.array(cList)
    contour_df = pd.DataFrame(cn).astype('float64')
    
    return view_df,contour_df


def suggest_exclude_view_list(view_df,view_thr_sd):
    """SUGGEST LIST OF VIEWS TO EXCLUDE BASED ON SD THRESHOLD"""
    # view mean/std = error in column 7
    view_mean = view_df.mean(axis=0)[7]
    view_std = view_df.std(axis=0)[7]
    view_thr = (view_thr_sd*view_std) + view_mean
    t = view_df.iloc[:][7] < view_thr
    to_exclude = [str(i+1) for i, x in enumerate(t) if not x]
    if to_exclude:
        suggested_views_to_exclude_str = ','.join(to_exclude)
        suggested_views_to_exclude = [int(i+1) for i, x in enumerate(t) if not x]
    else:
        to_exclude = '0'
        suggested_views_to_exclude_str = ','.join(to_exclude)
        suggested_views_to_exclude = [int(i+1) for i, x in enumerate(t) if not x]

    return suggested_views_to_exclude,suggested_views_to_exclude_str

 
def suggest_exclude_contour_list(contour_df,contour_thr_sd):
    """SUGGEST LIST OF CONTOURS TO EXCLUDE BASED ON SD THRESHOLD"""
    # contour mean/std = error in column 6
    contour_mean = contour_df.mean(axis=0)[6]
    contour_std = contour_df.std(axis=0)[6]
    contour_thr = (contour_thr_sd*contour_std) + contour_mean
    t = contour_df.iloc[:][6] < contour_thr
    suggested_contours_to_exclude = [float(i+1) for i, x in enumerate(t) if not x]
    
    return suggested_contours_to_exclude


def prune_fiducial_file(fiducial_file_df,suggested_contours_to_exclude):
    """EDIT FIDUCIAL DF"""
    logical_cont = fiducial_file_df[1].isin(suggested_contours_to_exclude)
    indx_log = [i for i, x in enumerate(logical_cont) if x]
    fid2 = fiducial_file_df.copy()
    pruned_fiducial_file = fid2.drop(indx_log)
    
    return pruned_fiducial_file


def makealigncom_excludeList(aligncom,aligncomout,excludelist):
    """MAKE ALIGN COM WITH EXCLUDE LIST"""
    global cc
    with open(aligncom, 'r') as file:
        data = file.readlines()
        c = 0
        for line in data:
            if line.startswith('LocalSkewDefaultGrouping'):
                cc = c
                if excludelist != '0':
                    data.insert(cc+1,f'ExcludeList\t{excludelist}\n')
                else:
                    print("No exclusions")
                    continue
            c+=1
    with open(aligncomout, 'w', encoding='utf-8') as file:
        file.writelines(data)


def maketiltcom_excludeList(tiltcom,tiltcomout,excludelist):
    """MAKE TILT COM WITH EXCLUDE LIST"""
    global cc
    with open(tiltcom, 'r') as file:
        data = file.readlines()
        c = 0
        for line in data:
            if line.startswith('IMAGEBINNED'):
                cc = c
                if excludelist != '0':
                    data.insert(cc+1,f'EXCLUDELIST\t{excludelist}\n')
                else: 
                    print("No exclusions")
                    continue
            c+=1
    with open(tiltcomout, 'w', encoding='utf-8') as file:
        file.writelines(data)
    trimmed_fidfile = ''
    fidlog = ''
    return trimmed_fidfile,fidlog


def check_alignlog(path_to_alignlog:str) -> bool:
    """CHECK ALIGN LOG EXISTS AND WORKED"""
    return 1 if os.path.exists(path_to_alignlog) and os.path.getsize(path_to_alignlog) > 10000 else 0


def check_etomo_state(file_name:str) -> str:
    """CHECK THE ETOMO BATCH LOG FILE AND SEE IF IT FINISHED SUCCESSFULLY
    """
    if not os.path.exists(file_name):
        #print('log file not created yet, waiting')
        return 0

    with open(file_name, 'r') as f:
        lines = f.readlines()
        last_row = 1
        while last_row:
            if lines[last_row*-1] == '\n':
                last_row +=1
            else:
                return(lines[last_row*-1].split()[0])

####### FUNCTION BLOCK ABOVE #######
####################################

def optimization(ts:str) -> str:
    """check if final optimized file exists"""
    file_to_check = f'{ts}/align_noRot2.log'
    alignlogexists = check_alignlog(file_to_check)
    if alignlogexists == 1:
        return(f'File exists and succeeded, skip tomogram {ts}.')
    else:
        print(f'Not optimized yet. Start working on {ts}.')

    os.chdir(ts)
    try:
        print(f'Waiting for etomo {ts} job to finish...')
        while True:
            batch_state = check_etomo_state('batchruntomo.log')
            if batch_state == 'ABORT':
                print(f'Warning! The batchetomo job failed for {ts}. Skip this tomogram.')
                raise Exception
            elif batch_state == 'Completed':
                print(f'The etomo {ts} job completed. Continue...')
                break
            else:
                time.sleep(60)

        print(f'Optimizing alignment for {ts} :')
        # Alignment without axis offset
        print('Alignment without axis offset:')
        makealigncom(f'align.com',f'align_noRot.com')
        makenewstcom(f'newst.com',f'newst_noRot.com')
        maketiltcom(f'tilt.com',f'tilt_noRot.com')
        submfg('align_noRot.com')
        submfg('newst_noRot.com')
        submfg('tilt_noRot.com')

        # prune contours, repeat alignment, newstack and reconstruction
        print('Pruning views and contours:')
        subprocess.run(f'model2point -c -ob -fl -inp {ts}.fid -ou {ts}_fid.pt', shell=True, check=True,stdout=subprocess.DEVNULL)
        fiducial_file_df = read_fiducial_file(f'{ts}_fid.pt')
        view_df,contour_df = read_align_log('align_noRot.log')
        suggested_views_to_exclude,suggested_views_to_exclude_str = suggest_exclude_view_list(view_df,view_thr_sd)
        suggested_contours_to_exclude = suggest_exclude_contour_list(contour_df,contour_thr_sd)
        pruned_fiducial_file = prune_fiducial_file(fiducial_file_df,suggested_contours_to_exclude)
        prepare_fiducial_file_to_write(pruned_fiducial_file,f'{ts}_fidPrune.pt')
        subprocess.run(f'mv {ts}.fid {ts}_bk.fid', shell=True, check=True)
        subprocess.run(f'point2model -op -ci 5 -w 2 -co 157,0,255 -zs 3 -im {ts}.preali -in {ts}_fidPrune.pt -ou {ts}.fid', shell=True, check=True)
        makealigncom_excludeList('align_noRot.com','align_clean.com',suggested_views_to_exclude_str)
        maketiltcom_excludeList('tilt_noRot.com','tilt_clean.com',suggested_views_to_exclude_str)
        submfg('align_clean.com')
        submfg('newst_noRot.com')
        submfg('tilt_clean.com')

        # final rotation
        if os.path.exists(f'{ts}.rec'):
            print(f'start of rotation on {ts}.rec')
            subprocess.run(f'trimvol -rx {ts}.rec {ts}_rot.mrc', shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f'clip flipz {ts}_rot.mrc {ts}_rot_flipz.mrc', shell=True, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(f'rm -f {ts}_rot.mrc', shell=True)
            print(f'end of rotation on {ts}.rec')
        else:
            print(f'{ts}.rec does not exist for whatever reason. Check back and see what went wrong!')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        os.chdir('../')
        if os.path.exists(f'{ts}/{ts}_rot_flipz.mrc'):
            return f"{ts} optimization is done, rotated tomogram saved as {ts}/{ts}_rot_flipz.mrc"
        else:
            return f"{ts} optimization encountered issue. {ts}/{ts}_rot_flipz.mrc not available"


def run_alignment():
    """Main function to run the eTomo alignment process."""
    tomo_list, etomo_com = make_etomo_files()

    print("Starting batchetomo process...")
    submfg(etomo_com)
    print("Batchetomo process finished.")

    # print("Starting parallel optimization for all tomograms...")
    # with Pool(cfg.etomo_cpu_cores) as pool:
    #     checkmarks = [pool.apply_async(optimization, args=(ts,)) for ts in tomo_list]
    #     for checkmark in checkmarks:
    #         print(f'{checkmark.get()}')
            
    print("All tasks are done.")
