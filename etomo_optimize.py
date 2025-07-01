#!/usr/bin/env python3
#
# Wrapper for batchtomo
#
# This script is designed to be called by a pipeline manager. It iterates
# through tomogram directories, identifies high-residual views and contours,
# and re-runs alignment and reconstruction steps to optimize the output.
#
# Refactoring (July 2025):
# - Encapsulated the logic for processing a single tomogram into an
#   `eTomoOptimizer` class to improve structure, maintainability, and safety.
# - Removed `os.chdir` calls, making path management more robust.
# - Centralized configuration and hardcoded values into constants.
# - Improved function/method names, typing, and documentation.
# - Enhanced error handling and logging within the class structure.
# - Made log file parsing robust by reading headers instead of using fixed column indices.

import glob
import os
import pandas as pd
import subprocess
from multiprocessing import Pool
import logging
import sys
from typing import List, Tuple, Iterator

# --- Configuration and Constants ---
try:
    import config as cfg
    NUM_CPU_CORES = cfg.etomo_cpu_cores
    TOMO_MATCH_STRING = cfg.tomo_match_string
    FINAL_X_SIZE = cfg.final_x_size
    FINAL_Y_SIZE = cfg.final_y_size
    THICKNESS_PXL = cfg.thickness_pxl
    FINAL_NEWSTACK_BIN = cfg.FINAL_NEWSTACK_BIN
except ImportError:
    logging.warning("Could not import config.py. Using fallback default values.")
    NUM_CPU_CORES = 8
    TOMO_MATCH_STRING = "L_"
    FINAL_X_SIZE = 512
    FINAL_Y_SIZE = 512
    THICKNESS_PXL = 3000
    FINAL_NEWSTACK_BIN = 8

os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_CPU_CORES)

VIEW_THR_SD = 2
CONTOUR_THR_SD = 2
SIRT_ITERATIONS = 20

class LogParsingError(Exception):
    """Custom exception for errors during log file parsing."""
    pass

# --- Helper Functions (File I/O) ---

def read_align_log(path_to_align_log: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads an IMOD align log and separates it into DataFrames for views and contours."""
    with open(path_to_align_log, 'r') as file:
        lines = file.readlines()

    def parse_section(lines_iterator: Iterator[str], expected_tokens: List[str]) -> pd.DataFrame:
        """
        Finds a section by matching the first few tokens of its header line,
        making it robust against whitespace changes.
        """
        header = None
        data_lines = []
        
        # Find the header line by matching tokens
        for line in lines_iterator:
            tokens = line.strip().split()
            if len(tokens) >= len(expected_tokens) and all(
                tokens[i] == expected_tokens[i] for i in range(len(expected_tokens))
            ):
                header = line.strip().replace('#', 'point_num').split()
                break
        
        if not header:
            raise LogParsingError(f"Header with starting tokens {expected_tokens} not found in log file.")
            
        # Collect data lines until a blank line is encountered
        for line in lines_iterator:
            if not line.strip():
                break
            data_lines.append(line.split())
            
        if not data_lines:
            logging.warning(f"Found header for {expected_tokens} but no data rows followed.")
            return pd.DataFrame(columns=header)
            
        df = pd.DataFrame(data_lines, columns=header)
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where key columns could not be parsed, as they are unusable
        if 'resid-nm' in df.columns:
            df.dropna(subset=['resid-nm'], inplace=True)
        return df

    # Use lists of tokens for robust header matching
    view_df = parse_section(iter(lines), ['view', 'rotation', 'tilt'])
    contour_df = parse_section(iter(lines), ['#', 'X', 'Y'])
    
    return view_df, contour_df


def read_fiducial_file(path_to_fiducial_file: str) -> pd.DataFrame:
    """Reads a fiducial point file into a pandas DataFrame."""
    # The columns are: object, contour, x, y, z
    return pd.read_csv(
        path_to_fiducial_file,
        sep=r'\s+',
        header=None,
        names=['object', 'contour', 'x', 'y', 'z'],
        engine='python'
    )

# --- Main Optimizer Class ---

class eTomoOptimizer:
    """
    Manages the optimization process for a single tomogram series.
    """
    def __init__(self, ts_directory: str):
        self.ts_dir = os.path.abspath(ts_directory)
        self.ts_name = os.path.basename(self.ts_dir)
        self.logger = self._setup_logger()

        self.align_log_path = os.path.join(self.ts_dir, 'align.log')
        self.final_mrc_path = os.path.join(self.ts_dir, f'{self.ts_name}_rot_flipz.mrc')

    def _setup_logger(self) -> logging.Logger:
        """Sets up a dedicated logger for this tomogram instance."""
        logger = logging.getLogger(self.ts_name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_path = os.path.join(self.ts_dir, 'optimization.log')
            fh = logging.FileHandler(log_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.propagate = False
        return logger

    def _run_command(self, command: List[str], log_msg: str) -> None:
        """Runs a command using subprocess and logs the output."""
        self.logger.info(log_msg)
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.ts_dir
            )
            self.logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
            self.logger.error(e.stderr)
            raise

    def _analyze_and_prune(self) -> Tuple[str, bool]:
        """
        Analyzes alignment logs, suggests exclusions, and returns the view
        exclusion list and a boolean indicating if re-alignment is needed.
        """
        view_df, contour_df = read_align_log(self.align_log_path)

        if view_df.empty:
            self.logger.warning("View data is empty after parsing. Cannot exclude views.")
        if contour_df.empty:
            self.logger.warning("Contour data is empty after parsing. Cannot exclude contours.")

        # Suggest views to exclude based on residual mean and standard deviation
        views_to_exclude = []
        if 'resid-nm' in view_df.columns and not view_df.empty:
            view_mean = view_df['resid-nm'].mean()
            view_std = view_df['resid-nm'].std()
            view_thr = (VIEW_THR_SD * view_std) + view_mean
            views_to_exclude_df = view_df[view_df['resid-nm'] > view_thr]
            views_to_exclude = views_to_exclude_df['view'].astype(int).astype(str).tolist()
        views_to_exclude_str = ','.join(views_to_exclude) if views_to_exclude else '0'

        # Suggest contours to exclude based on residual mean and standard deviation
        contours_to_exclude = []
        if 'resid-nm' in contour_df.columns and not contour_df.empty:
            contour_mean = contour_df['resid-nm'].mean()
            contour_std = contour_df['resid-nm'].std()
            contour_thr = (CONTOUR_THR_SD * contour_std) + contour_mean
            contours_to_exclude_df = contour_df[contour_df['resid-nm'] > contour_thr]
            contours_to_exclude = contours_to_exclude_df['cont'].astype(int).unique().tolist()

        realign_needed = bool(views_to_exclude or contours_to_exclude)
        if not realign_needed:
            self.logger.info("No new views or contours to exclude. Skipping re-alignment.")
            return '0', False

        self.logger.info(f'Pruning: {len(views_to_exclude)} views and {len(contours_to_exclude)} contours to exclude.')
        
        # Prune fiducial file if necessary
        if contours_to_exclude:
            fid_pt_path = os.path.join(self.ts_dir, f'{self.ts_name}_fid.pt')
            self._run_command(['model2point', '-c', '-ob', '-inp', f'{self.ts_name}.fid', '-ou', fid_pt_path], "Converting .fid to .pt")
            
            fid_df = read_fiducial_file(fid_pt_path)
            
            logical_cont = fid_df['contour'].isin(contours_to_exclude)
            pruned_fid_df = fid_df.drop(fid_df[logical_cont].index)

            pruned_fid_pt_path = os.path.join(self.ts_dir, f'{self.ts_name}_fidPrune.pt')
            
            # Ensure correct types before saving
            for col in ['point', 'contour']:
                if col in pruned_fid_df:
                    pruned_fid_df[col] = pruned_fid_df[col].astype('int')

            # Select only the columns that point2model needs
            output_df = pruned_fid_df[['object', 'contour', 'x', 'y', 'z']]
            output_df.to_csv(pruned_fid_pt_path, index=False, header=False, sep='\t')

            # Backup original and create new .fid file
            if not os.path.exists(os.path.join(self.ts_dir, f'{self.ts_name}_bk.fid')):
                os.rename(os.path.join(self.ts_dir, f'{self.ts_name}.fid'), os.path.join(self.ts_dir, f'{self.ts_name}_bk.fid'))
            
            self._run_command([
                'point2model', '-op', '-ci', '5', '-w', '2', '-co', '157,0,255', '-zs', '3',
                '-im', f'{self.ts_name}_preali.mrc', '-in', pruned_fid_pt_path, '-ou', f'{self.ts_name}.fid'
            ], "Creating new .fid from pruned points")

        return views_to_exclude_str, True

    def _create_command_files(self, views_to_exclude_str: str, realign_needed: bool):
        """Creates the necessary .com files for IMOD."""
        # Create newst_clean.com
        with open(os.path.join(self.ts_dir, 'newst.com'), 'r') as f_in, \
             open(os.path.join(self.ts_dir, 'newst_clean.com'), 'w') as f_out:
            for line in f_in:
                if line.startswith('SizeToOutputInXandY'):
                    f_out.write(f"SizeToOutputInXandY\t{FINAL_X_SIZE},{FINAL_Y_SIZE}\n")
                else:
                    f_out.write(line)

        # Create tilt_clean.com
        with open(os.path.join(self.ts_dir, 'tilt.com'), 'r') as f_in, \
             open(os.path.join(self.ts_dir, 'tilt_clean.com'), 'w') as f_out:
            lines = f_in.readlines()
            sirt_added = any('FakeSIRTiterations' in line for line in lines)
            for line in lines:
                if line.startswith('THICKNESS'):
                    f_out.write(f'THICKNESS\t{int(THICKNESS_PXL / FINAL_NEWSTACK_BIN)}\n')
                elif line.startswith('TILTFILE') and views_to_exclude_str != '0':
                    f_out.write(f'EXCLUDELIST\t{views_to_exclude_str}\n')
                    f_out.write(line)
                elif line.startswith('InputProjections') and not sirt_added:
                    f_out.write(f'FakeSIRTiterations\t{SIRT_ITERATIONS}\n')
                    f_out.write(line)
                    sirt_added = True
                else:
                    f_out.write(line)

        # Create align_clean.com if needed
        if realign_needed:
            with open(os.path.join(self.ts_dir, 'align.com'), 'r') as f_in, \
                 open(os.path.join(self.ts_dir, 'align_clean.com'), 'w') as f_out:
                for line in f_in:
                    f_out.write(line)
                    if line.startswith('LocalSkewDefaultGrouping') and views_to_exclude_str != '0':
                        f_out.write(f'ExcludeList\t{views_to_exclude_str}\n')

    def run(self) -> str:
        """Executes the full optimization pipeline for the tomogram."""
        if os.path.exists(self.final_mrc_path):
            self.logger.info(f'Final file already exists. Skipping.')
            return f'Skipped {self.ts_name}: Final file already exists.'

        self.logger.info('Starting optimization.')
        if not os.path.exists(self.align_log_path):
            return f"FAILED {self.ts_name}: align.log not found."

        try:
            views_to_exclude_str, realign_needed = self._analyze_and_prune()
            self._create_command_files(views_to_exclude_str, realign_needed)

            if realign_needed:
                self._run_command(['submfg', 'align_clean.com'], 'Running cleaned alignment...')
            
            self.logger.info('Running cleaned newstack and tilt...')
            self._run_command(['submfg', 'newst_clean.com'], 'Running newstack...')
            self._run_command(['submfg', 'tilt_clean.com'], 'Running tilt...')

            rec_file = os.path.join(self.ts_dir, f'{self.ts_name}_rec.mrc')
            if os.path.exists(rec_file):
                self.logger.info(f'Starting final rotation on {rec_file}.')
                rot_file = os.path.join(self.ts_dir, f'{self.ts_name}_rot.mrc')
                self._run_command(['trimvol', '-rx', rec_file, rot_file], "Rotating tomogram...")
                self._run_command(['clip', 'flipz', rot_file, self.final_mrc_path], "Flipping Z-axis...")
                os.remove(rot_file)
                self.logger.info('Finished final rotation.')
            else:
                self.logger.warning(f'{rec_file} does not exist. Cannot perform final rotation.')

        except (LogParsingError, ValueError) as e:
            self.logger.error(f"Analysis failed due to a parsing or value error: {e}", exc_info=True)
            return f"{self.ts_name}: Optimization FAILED. Reason: {e}"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return f"{self.ts_name}: Optimization FAILED. Reason: {e}"
        finally:
            logging.shutdown()

        if os.path.exists(self.final_mrc_path):
            return f"{self.ts_name}: Optimization complete. Rotated tomogram saved."
        else:
            return f"{self.ts_name}: Optimization may have failed. Final tomogram not created."


def optimization_worker(ts_directory: str) -> str:
    """
    Worker function for multiprocessing.Pool.
    Instantiates and runs the eTomoOptimizer for a given tomogram directory.
    """
    optimizer = eTomoOptimizer(ts_directory)
    return optimizer.run()


def run_optimization_pipeline():
    """Main function to find tomograms and run the optimization in parallel."""
    main_logger = logging.getLogger(__name__)
    
    try:
        tomo_list = sorted(glob.glob(f'{TOMO_MATCH_STRING}*'))
        if not tomo_list:
            main_logger.error(f"No tomogram directories found matching pattern: {TOMO_MATCH_STRING}*")
            return
    except Exception as e:
        main_logger.error(f"Error finding tomograms: {e}")
        return

    main_logger.info(f"Starting parallel optimization for {len(tomo_list)} tomograms...")
    
    with Pool(NUM_CPU_CORES) as pool:
        results = pool.map(optimization_worker, tomo_list)
        for res in results:
            main_logger.info(res)
            
    main_logger.info("All optimization tasks are complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    run_optimization_pipeline()
