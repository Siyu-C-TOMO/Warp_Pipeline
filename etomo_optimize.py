#!/usr/bin/env python3
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
import pandas as pd
import numpy as np
import config as cfg
import subprocess
from multiprocessing import Pool
import logging
import sys

os.environ['NUMEXPR_MAX_THREADS'] = str(cfg.etomo_cpu_cores)

# thresholds for pruning views and contours
view_thr_sd = 2
contour_thr_sd = 2


####################################
####### FUNCTION BLOCK BELOW #######

def list_tomo() -> list[str]:
    """List all tomograms in the current directory."""
    pattern = f'{cfg.tomo_match_string}*'
    tomo_list = sorted(glob.glob(pattern))
    if not tomo_list:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return tomo_list

def submfg(file_name, logger) -> None:
    """Submit a command file to etomo using submfg."""
    logger.info(f"Starting Command {file_name} using etomo")
    try:
        result = subprocess.run(['submfg', file_name], check=True, capture_output=True, text=True)
        logger.info(f"Command {file_name} ran successfully.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command {file_name} failed with return code {e.returncode}")
        logger.error(e.stderr)
        raise

def makenewstcom(newstcom, newstcomout):
    """Make new stack command with correct final size, using values from config."""
    size_string = f"SizeToOutputInXandY\t{cfg.final_x_size},{cfg.final_y_size}\n"
    
    with open(newstcom, 'r') as file:
        data = file.readlines()

    for i, line in enumerate(data):
        if line.startswith('SizeToOutputInXandY'):
            data[i] = size_string
            break
    
    with open(newstcomout, 'w', encoding='utf-8') as file:
        file.writelines(data)

def make_clean_tiltcom(tiltcom_in, tiltcom_out, excludelist):
    """Make tilt command to reconstruct the tomogram with SIRTlike iterations,
    apply the correct thickness, and add an exclude list."""
    SIRT_set = 'FakeSIRTiterations\t20\n'
    thickness_set = f'THICKNESS\t{int(cfg.thickness_pxl/cfg.FINAL_NEWSTACK_BIN)}\n'
    
    with open(tiltcom_in, 'r') as file:
        original_lines = file.readlines()

    new_lines = []
    sirt_added = False
    thickness_replaced = False
    exclude_added = False

    for line in original_lines:
        if line.startswith('THICKNESS') and not thickness_replaced:
            new_lines.append(thickness_set)
            thickness_replaced = True
        elif line.startswith('TILTFILE') and not exclude_added:
            if excludelist != '0':
                new_lines.append(f'EXCLUDELIST\t{excludelist}\n')
            exclude_added = True
            new_lines.append(line)
        elif line.startswith('InputProjections') and not sirt_added:
            new_lines.append(SIRT_set)
            new_lines.append(line)
            sirt_added = True
        else:
            new_lines.append(line)

    with open(tiltcom_out, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

def read_fiducial_file(path_to_fiducial_file):
    """READ FIDUCIAL FILE AS CSV FOR FURTHER PROCESSING
    """
    fiducial_file_df = pd.read_csv(
        path_to_fiducial_file,
        sep=r'\s+',
        header=None,
        engine='python'
    )
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


def makealigncom_excludeList(aligncom, aligncomout, excludelist, logger):
    """MAKE ALIGN COM WITH EXCLUDE LIST"""
    with open(aligncom, 'r') as file:
        data = file.readlines()
    
    insert_pos = -1
    for i, line in enumerate(data):
        if line.startswith('LocalSkewDefaultGrouping'):
            insert_pos = i + 1
            break
    
    if insert_pos != -1:
        if excludelist and excludelist != '0':
            logger.info(f"Adding ExcludeList to {aligncomout}")
            data.insert(insert_pos, f'ExcludeList\t{excludelist}\n')
        else:
            logger.info("No views to exclude.")
    else:
        logger.warning("Could not find 'LocalSkewDefaultGrouping' in align.com. ExcludeList not added.")

    with open(aligncomout, 'w', encoding='utf-8') as file:
        file.writelines(data)

def check_alignlog(path_to_alignlog:str) -> bool:
    """CHECK ALIGN LOG EXISTS AND WORKED"""
    return 1 if os.path.exists(path_to_alignlog) and os.path.getsize(path_to_alignlog) > 10000 else 0


def check_etomo_state(file_name:str) -> str:
    """CHECK THE ETOMO BATCH LOG FILE AND SEE IF IT FINISHED SUCCESSFULLY
    """
    if not os.path.exists(file_name):
        #logging.debug('log file not created yet, waiting')
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

def optimization(ts: str) -> str:
    """
    Performs alignment optimization for a single tomogram series.
    This function is designed to be called by multiprocessing.Pool,
    so it changes directory into the tomogram folder and sets up its own logger.
    """
    original_dir = os.getcwd()
    os.chdir(ts)

    logger = logging.getLogger(ts)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('optimization.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False

    try:
        final_file = f'{ts}_rot_flipz.mrc'
        if os.path.exists(final_file):
            logger.info(f'Final file {final_file} already exists. Skipping.')
            return f'Skipped {ts}: Final file already exists.'

        logger.info('Starting optimization.')

        if not os.path.exists('align.log'):
            raise FileNotFoundError("align.log not found. Cannot proceed with optimization.")

        # --- Step 1: Analyze logs and decide on exclusions ---
        subprocess.run(f'model2point -c -ob -inp {ts}.fid -ou {ts}_fid.pt', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fiducial_file_df = read_fiducial_file(f'{ts}_fid.pt')
        view_df, contour_df = read_align_log('align.log')
        suggested_views_to_exclude, suggested_views_to_exclude_str = suggest_exclude_view_list(view_df, view_thr_sd)
        suggested_contours_to_exclude = suggest_exclude_contour_list(contour_df, contour_thr_sd)

        logger.info("Creating cleaned newst.com and tilt.com with optimized parameters.")
        makenewstcom('newst.com', 'newst_clean.com')
        make_clean_tiltcom('tilt.com', 'tilt_clean.com', suggested_views_to_exclude_str)

        if not suggested_views_to_exclude and not suggested_contours_to_exclude:
            logger.info("No new views or contours to exclude. Skipping re-alignment step.")
        else:
            logger.info(f'Pruning: {len(suggested_views_to_exclude)} views and {len(suggested_contours_to_exclude)} contours to exclude.')
            pruned_fiducial_file = prune_fiducial_file(fiducial_file_df, suggested_contours_to_exclude)
            
            prepare_fiducial_file_to_write(pruned_fiducial_file, f'{ts}_fidPrune.pt')
            if not os.path.exists(f'{ts}_bk.fid'):
                subprocess.run(f'mv {ts}.fid {ts}_bk.fid', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Redirect point2model output to the log file
            with open('optimization.log', 'a') as log_file:
                subprocess.run(f'point2model -op -ci 5 -w 2 -co 157,0,255 -zs 3 -im {ts}_preali.mrc -in {ts}_fidPrune.pt -ou {ts}.fid', shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            
            logger.info("Creating cleaned align.com.")
            makealigncom_excludeList('align.com', 'align_clean.com', suggested_views_to_exclude_str, logger)

            logger.info('Running cleaned alignment...')
            submfg('align_clean.com', logger)

        logger.info('Running cleaned newstack and tilt...')
        submfg('newst_clean.com', logger)
        submfg('tilt_clean.com', logger)

        if os.path.exists(f'{ts}_rec.mrc'):
            logger.info(f'Starting final rotation on {ts}_rec.mrc.')
            subprocess.run(f'trimvol -rx {ts}_rec.mrc {ts}_rot.mrc', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(f'clip flipz {ts}_rot.mrc {ts}_rot_flipz.mrc', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(f'rm -f {ts}_rot.mrc', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info('Finished final rotation.')
        else:
            logger.warning(f'{ts}_rec.mrc does not exist. Cannot perform final rotation.')

    except Exception as e:
        logger.error(f"An error occurred during optimization for {ts}: {e}", exc_info=True)
        # Try to capture the last few lines of the most recent log file for a summary
        error_summary = str(e)
        try:
            with open('align_clean.log', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    error_summary = f"Error after/during align_clean.com. Last log line: {last_line}"
        except FileNotFoundError:
            pass
        return f"{ts}: Optimization FAILED. Reason: {error_summary}"
    finally:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        os.chdir(original_dir)

    if os.path.exists(f'{ts}/{ts}_rot_flipz.mrc'):
        return f"{ts}: Optimization complete. Rotated tomogram saved."
    else:
        return f"{ts}: Optimization may have failed. Final tomogram not created."


def run_optimization():
    """Main function to run the eTomo optimization process."""   
    logger = logging.getLogger(__name__)

    try:
        tomo_list = list_tomo()
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"Starting parallel optimization for {len(tomo_list)} tomograms...")
    
    with Pool(cfg.etomo_cpu_cores) as pool:
        results = [pool.apply_async(optimization, args=(ts,)) for ts in tomo_list]
        for res in results:
            logger.info(res.get())
            
    logger.info("All optimization tasks are complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    run_optimization()
