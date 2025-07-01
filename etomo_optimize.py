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
# - Replaced os.path with pathlib for modern, object-oriented path handling.
# - Centralized configuration and hardcoded values into constants.
# - Improved function/method names, typing, and documentation.
# - Enhanced error handling and logging within the class structure.
# - Made log file parsing robust by reading headers instead of using fixed column indices.

import os
import pandas as pd
import subprocess
from multiprocessing import Pool
import logging
import sys
from typing import List, Tuple, Iterator, Union, Dict
from pathlib import Path

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

def parse_section(lines_iterator: Iterator[str], expected_tokens: List[str]) -> pd.DataFrame:
    """Finds and parses a specific data section from the log file lines."""
    header = None
    data_lines = []
    
    for line in lines_iterator:
        tokens = line.strip().split()
        if len(tokens) >= len(expected_tokens) and all(
            tokens[i] == expected_tokens[i] for i in range(len(expected_tokens))
        ):
            header = line.strip().replace('#', 'point_num').split()
            break
    
    if not header:
        raise LogParsingError(f"Header with starting tokens {expected_tokens} not found.")
        
    for line in lines_iterator:
        if not line.strip():
            break
        data_lines.append(line.split())
        
    if not data_lines:
        logging.warning(f"Found header for {expected_tokens} but no data rows followed.")
        return pd.DataFrame(columns=header)
        
    df = pd.DataFrame(data_lines, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'resid-nm' in df.columns:
        df.dropna(subset=['resid-nm'], inplace=True)
    return df

def read_align_log(path_to_align_log: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads an IMOD align log and separates it into DataFrames for views and contours."""
    with path_to_align_log.open('r') as f:
        lines = f.readlines()

    view_df = parse_section(iter(lines), ['view', 'rotation', 'tilt'])
    contour_df = parse_section(iter(lines), ['#', 'X', 'Y'])
    
    return view_df, contour_df


def read_fiducial_file(path_to_fiducial_file: Path) -> pd.DataFrame:
    """Reads a fiducial point file into a pandas DataFrame."""
    return pd.read_csv(
        path_to_fiducial_file,
        sep=r'\s+',
        header=None,
        names=['object', 'contour', 'x', 'y', 'z'],
        engine='python'
    )

# --- Main Optimizer Class ---

class eTomoOptimizer:
    """Manages the optimization process for a single tomogram series."""
    def __init__(self, ts_directory: str):
        self.ts_dir = Path(ts_directory).resolve()
        self.ts_name = self.ts_dir.name
        self.logger = self._setup_logger()

        self.align_log_path = self.ts_dir / 'align.log'
        self.final_mrc_path = self.ts_dir / f'{self.ts_name}_rot_flipz.mrc'

    def _setup_logger(self) -> logging.Logger:
        """Sets up a dedicated logger for this tomogram instance."""
        logger = logging.getLogger(self.ts_name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_path = self.ts_dir / 'optimization.log'
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

    def _identify_outliers(self, df: pd.DataFrame, id_col: str, threshold_sd: float) -> List[Union[str, int]]:
        """Identifies outliers in a DataFrame based on the standard deviation of 'resid-nm'."""
        if df.empty or 'resid-nm' not in df.columns:
            return []
        
        mean = df['resid-nm'].mean()
        std_dev = df['resid-nm'].std()
        threshold = (threshold_sd * std_dev) + mean
        
        outliers_df = df[df['resid-nm'] > threshold]
        return outliers_df[id_col].unique().tolist()

    def _analyze_alignment_logs(self) -> Tuple[List[str], List[int]]:
        """
        Analyzes alignment logs to identify views and contours to exclude.
        Returns a tuple of (views_to_exclude, contours_to_exclude).
        """
        view_df, contour_df = read_align_log(self.align_log_path)

        views_to_exclude = [str(int(v)) for v in self._identify_outliers(view_df, 'view', VIEW_THR_SD)]
        contours_to_exclude = [int(c) for c in self._identify_outliers(contour_df, 'cont', CONTOUR_THR_SD)]
        
        return views_to_exclude, contours_to_exclude

    def _prune_fiducial_model(self, contours_to_exclude: List[int]):
        """
        Creates a new, pruned fiducial model if contours are excluded.
        This method has side effects: it backs up and overwrites the original .fid file.
        """
        if not contours_to_exclude:
            return

        self.logger.info(f"Pruning {len(contours_to_exclude)} contours from fiducial model.")
        
        fid_pt_path = self.ts_dir / f'{self.ts_name}_fid.pt'
        self._run_command(['model2point', '-c', '-ob', '-inp', f'{self.ts_name}.fid', '-ou', str(fid_pt_path)], "Converting .fid to .pt")
        
        fid_df = read_fiducial_file(fid_pt_path)
        pruned_fid_df = fid_df[~fid_df['contour'].isin(contours_to_exclude)]
        pruned_fid_pt_path = self.ts_dir / f'{self.ts_name}_fidPrune.pt'
        
        output_df = pruned_fid_df[['object', 'contour', 'x', 'y', 'z']]
        output_df.to_csv(pruned_fid_pt_path, index=False, header=False, sep='\t')

        original_fid = self.ts_dir / f'{self.ts_name}.fid'
        backup_fid = self.ts_dir / f'{self.ts_name}_bk.fid'
        if not backup_fid.exists():
            original_fid.rename(backup_fid)
        
        self._run_command([
            'point2model', '-op', '-ci', '5', '-w', '2', '-co', '157,0,255', '-zs', '3',
            '-im', f'{self.ts_name}_preali.mrc', '-in', str(pruned_fid_pt_path), '-ou', str(original_fid)
        ], "Creating new .fid from pruned points")

    def _update_com_file(self, source_path: Path, dest_path: Path, modifications: Dict[str, Dict[str, str]]):
        """
        Reads a source .com file, applies modifications, and writes to a destination file.
        It replaces existing parameters or inserts them after a specified anchor if they don't exist.
        """
        if not source_path.exists():
            self.logger.error(f"Source .com file not found: {source_path}")
            return
        source_lines = source_path.read_text().splitlines()
        rules_to_process = modifications.copy()
        
        lines_after_replacement = []
        for line in source_lines:
            key = line.split('\t')[0].strip()
            if key in rules_to_process:
                lines_after_replacement.append(f"{key}\t{rules_to_process[key]['value']}")
                del rules_to_process[key]
            else:
                lines_after_replacement.append(line)
        
        if not rules_to_process:
            final_lines = lines_after_replacement
        else:
            lines_after_insertion = []
            keys_to_insert = set(rules_to_process.keys())
            
            for line in lines_after_replacement:
                lines_after_insertion.append(line)
                current_line_key = line.split('\t')[0].strip()
                
                for key in list(keys_to_insert):
                    if modifications[key]['anchor'] == current_line_key:
                        lines_after_insertion.append(f"{key}\t{modifications[key]['value']}")
                        keys_to_insert.remove(key)
            final_lines = lines_after_insertion

        dest_path.write_text('\n'.join(final_lines) + '\n')

    def _create_com_files(self, views_to_exclude_str: str, realign_needed: bool):
        """Creates the necessary .com files for IMOD using a unified update logic."""
        newst_mods = {
            'SizeToOutputInXandY': {
                'value': f'{FINAL_X_SIZE},{FINAL_Y_SIZE}',
                'anchor': 'AdjustOrigin'
            }
        }
        self._update_com_file(self.ts_dir / 'newst.com', self.ts_dir / 'newst_clean.com', newst_mods)

        tilt_mods = {
            'THICKNESS': {
                'value': str(int(THICKNESS_PXL / FINAL_NEWSTACK_BIN)),
                'anchor': 'XTILTFILE'
            },
            'FakeSIRTiterations': {
                'value': str(SIRT_ITERATIONS),
                'anchor': '$tilt'
            }
        }
        if views_to_exclude_str != '0':
            tilt_mods['EXCLUDELIST'] = {
                'value': views_to_exclude_str,
                'anchor': 'IMAGEBINNED'
            }
        self._update_com_file(self.ts_dir / 'tilt.com', self.ts_dir / 'tilt_clean.com', tilt_mods)

        if realign_needed:
            align_mods = {}
            if views_to_exclude_str != '0':
                align_mods['ExcludeList'] = {
                    'value': views_to_exclude_str,
                    'anchor': 'LocalSkewDefaultGrouping'
                }
            self._update_com_file(self.ts_dir / 'align.com', self.ts_dir / 'align_clean.com', align_mods)

    def _reconstruct(self):
        """Runs the reconstruction and finalization steps."""
        self.logger.info('Running cleaned newstack and tilt...')
        self._run_command(['submfg', 'newst_clean.com'], 'Running newstack...')
        self._run_command(['submfg', 'tilt_clean.com'], 'Running tilt...')

        rec_file = self.ts_dir / f'{self.ts_name}_rec.mrc'
        if rec_file.exists():
            self.logger.info(f'Starting final rotation on {rec_file}.')
            rot_file = self.ts_dir / f'{self.ts_name}_rot.mrc'
            self._run_command(['trimvol', '-rx', str(rec_file), str(rot_file)], "Rotating tomogram...")
            self._run_command(['clip', 'flipz', str(rot_file), str(self.final_mrc_path)], "Flipping Z-axis...")
            rot_file.unlink()
            self.logger.info('Finished final rotation.')
        else:
            self.logger.warning(f'{rec_file} does not exist. Cannot perform final rotation.')

    def run(self) -> str:
        """Executes the full optimization pipeline for the tomogram."""
        if self.final_mrc_path.exists():
            self.logger.info('Final file already exists. Skipping.')
            return f'Skipped {self.ts_name}: Final file already exists.'

        self.logger.info('Starting optimization.')
        if not self.align_log_path.exists():
            return f"FAILED {self.ts_name}: align.log not found."

        try:
            views_to_exclude, contours_to_exclude = self._analyze_alignment_logs()
            realign_needed = bool(views_to_exclude or contours_to_exclude)
            
            if realign_needed:
                self.logger.info(f'Pruning: {len(views_to_exclude)} views and {len(contours_to_exclude)} contours to exclude.')
                self._prune_fiducial_model(contours_to_exclude)
            else:
                self.logger.info("No new views or contours to exclude. Skipping re-alignment.")
            
            views_to_exclude_str = ','.join(views_to_exclude) if views_to_exclude else '0'
            self._create_com_files(views_to_exclude_str, realign_needed)

            if realign_needed:
                self._run_command(['submfg', 'align_clean.com'], 'Running cleaned alignment...')
            
            self._reconstruct()

        except (LogParsingError, ValueError) as e:
            self.logger.error(f"Analysis failed due to a parsing or value error: {e}", exc_info=True)
            return f"{self.ts_name}: Optimization FAILED. Reason: {e}"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return f"{self.ts_name}: Optimization FAILED. Reason: {e}"
        finally:
            logging.shutdown()

        if self.final_mrc_path.exists():
            return f"{self.ts_name}: Optimization complete. Rotated tomogram saved."
        else:
            return f"{self.ts_name}: Optimization may have failed. Final tomogram not created."


def optimization_worker(ts_directory: str) -> str:
    """
    Worker function for multiprocessing.Pool.
    Instantiates and runs the eTomoOptimizer for a given tomogram directory.
    """
    try:
        optimizer = eTomoOptimizer(ts_directory)
        return optimizer.run()
    except Exception as e:
        logging.error(f"Failed to initialize or run optimizer for {ts_directory}: {e}", exc_info=True)
        return f"FAILED {ts_directory}: {e}"


def run_optimization_pipeline():
    """Main function to find tomograms and run the optimization in parallel."""
    main_logger = logging.getLogger(__name__)
    
    try:
        tomo_list = sorted([str(p) for p in Path.cwd().glob(f'{TOMO_MATCH_STRING}*') if p.is_dir()])
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
