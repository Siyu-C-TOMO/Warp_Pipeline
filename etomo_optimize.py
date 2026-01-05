#!/usr/bin/env python3
import os
import pandas as pd
from multiprocessing import Pool
import logging
import sys
import argparse
from typing import List, Tuple, Union, Dict
from pathlib import Path
from functools import partial

from pipeline_utils import read_align_log, read_fiducial_file, LogParsingError, run_command
sys.path.insert(0, os.getcwd())

# --- Configuration and Constants ---
try:
    import config as cfg
    NUM_CPU_CORES = cfg.etomo_cpu_cores
    TOMO_MATCH_STRING = cfg.tomo_match_string
    FINAL_X_SIZE = cfg.final_x_size
    FINAL_Y_SIZE = cfg.final_y_size
    THICKNESS_PXL = cfg.thickness_pxl
    FINAL_NEWSTACK_BIN = cfg.FINAL_NEWSTACK_BIN
    CAMERA_TYPE = cfg.camera_type
except ImportError:
    logging.warning("Could not import config.py. Using fallback default values.")
    NUM_CPU_CORES = 8
    TOMO_MATCH_STRING = "L_"
    FINAL_X_SIZE = 512
    FINAL_Y_SIZE = 512
    THICKNESS_PXL = 3000
    FINAL_NEWSTACK_BIN = 8
    CAMERA_TYPE = "Falcon4"

os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_CPU_CORES)

VIEW_THR_SD = 3
CONTOUR_THR_SD = 2
SIRT_ITERATIONS = 20

# --- Main Optimizer Class ---

class eTomoOptimizer:
    """Manages the optimization process for a single tomogram series."""
    def __init__(self, ts_directory: str, tomogram_logs_dir: Path):
        self.ts_dir = Path(ts_directory).resolve()
        self.ts_name = self.ts_dir.name
        self.log_dir = tomogram_logs_dir / self.ts_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()

        self.align_log_path = self.ts_dir / 'align.log'
        self.final_mrc_path = self.ts_dir / f'{self.ts_name}_rot_flipz.mrc'

    def _setup_logger(self) -> logging.Logger:
        """Sets up a dedicated logger for this tomogram instance."""
        logger = logging.getLogger(self.ts_name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_path = self.log_dir / 'optimization.log'
            fh = logging.FileHandler(log_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.propagate = False
        return logger

    def _identify_outliers(self, df: pd.DataFrame, 
                           id_col: str, 
                           threshold_sd: float,
                           to_select: bool = False) -> List[Union[str, int]]:
        """Identifies outliers in a DataFrame based on the standard deviation of 'resid-nm'."""
        if df.empty or 'resid-nm' not in df.columns:
            return []
        
        mean = df['resid-nm'].mean()
        std_dev = df['resid-nm'].std()
        threshold = (threshold_sd * std_dev) + mean
        
        if to_select:
            inliers_df = df[df['resid-nm'] <= threshold]
            return inliers_df[id_col].unique().tolist()
        else:
            outliers_df = df[df['resid-nm'] > threshold]
            return outliers_df[id_col].unique().tolist()

    def _analyze_alignment_logs(self) -> Tuple[List[str], List[int]]:
        """
        Analyzes alignment logs to identify views and contours to exclude.
        Returns a tuple of (views_to_exclude, contours_to_exclude).
        """
        view_df, cont_df, bad_point_df = read_align_log(self.align_log_path)

        views_to_exclude = [str(int(v)) for v in self._identify_outliers(view_df, 'view', VIEW_THR_SD)]

        contours_to_include = [int(c) for c in self._identify_outliers(cont_df, 'cont', CONTOUR_THR_SD, to_select=True)]

        bad_contours = bad_point_df['cont'].dropna()
        if not bad_contours.empty:
            max_contour = bad_contours.max()
            contours_to_remove_set = {int(c) for c in bad_contours if c != max_contour and int(c) != 0}
            contours_to_include = [c for c in contours_to_include if c not in contours_to_remove_set]

        return views_to_exclude, contours_to_include

    def _prune_fiducial_model(self, contours_to_include: List[int]):
        """
        Creates a new, pruned fiducial model if contours are excluded.
        This method has side effects: it backs up and overwrites the original .fid file.
        """
        fid_pt_path = self.ts_dir / f'{self.ts_name}_fid.pt'
        cmd = ['model2point', '-c', '-ob', '-inp', f'{self.ts_name}.fid', '-ou', str(fid_pt_path)]
        run_command(cmd, self.log_dir / 'log_model2point.log', cwd=self.ts_dir)
            
        fid_df = read_fiducial_file(fid_pt_path)
        if contours_to_include:
            pruned_fid_df = fid_df[fid_df['contour'].isin(contours_to_include)]
        else:
            pruned_fid_df = fid_df.copy()            
        pruned_fid_pt_path = self.ts_dir / f'{self.ts_name}_fidPrune.pt'
        
        output_df = pruned_fid_df[['object', 'contour', 'x', 'y', 'z']]
        output_df.to_csv(pruned_fid_pt_path, index=False, header=False, sep='\t')

        original_fid = self.ts_dir / f'{self.ts_name}.fid'
        backup_fid = self.ts_dir / f'{self.ts_name}_bk.fid'
        if not backup_fid.exists():
            original_fid.rename(backup_fid)

        im_name = self.ts_name + '_preali.mrc' if (self.ts_dir / f'{self.ts_name}_preali.mrc').exists() else self.ts_name + '.preali'
        cmd = [
            'point2model', '-op', '-ci', '5', '-w', '2', '-co', '157,0,255', '-zs', '3',
            '-im', im_name, '-in', str(pruned_fid_pt_path), '-ou', str(original_fid)
        ]
        run_command(cmd, self.log_dir / 'log_point2model.log', cwd=self.ts_dir)

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
            key = line.strip().split()[0] if line.strip() else ""
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
                current_line_key = line.strip().split()[0] if line.strip() else ""
                
                for key in list(keys_to_insert):
                    if modifications[key]['anchor'] == current_line_key:
                        lines_after_insertion.append(f"{key}\t{modifications[key]['value']}")
                        keys_to_insert.remove(key)
            final_lines = lines_after_insertion

        dest_path.write_text('\n'.join(final_lines) + '\n')

    def _create_com_files(self, views_to_exclude_str: str):
        """Creates the necessary .com files for IMOD using a unified update logic."""
        newst_mods = {
            'SizeToOutputInXandY': {
                'value': f'{FINAL_Y_SIZE},{FINAL_X_SIZE}',
                'anchor': 'AdjustOrigin'
            }
        }
        self._update_com_file(self.ts_dir / 'newst.com', self.ts_dir / 'newst_clean.com', newst_mods)

        tilt_mods = {}
        if CAMERA_TYPE == "Falcon4":
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
        elif CAMERA_TYPE == "K3":
            tilt_mods = {
                'IMAGEBINNED': {
                    'value': str(int(FINAL_NEWSTACK_BIN)),
                    'anchor': 'OutputFile'
                }
            }
        if views_to_exclude_str != '0':
            tilt_mods['EXCLUDELIST'] = {
                'value': views_to_exclude_str,
                'anchor': 'IMAGEBINNED'
            }
        self._update_com_file(self.ts_dir / 'tilt.com', self.ts_dir / 'tilt_clean.com', tilt_mods)

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
        run_command(['submfg', 'newst_clean.com'], self.log_dir / 'log_newst.log', cwd=self.ts_dir)
        run_command(['submfg', 'tilt_clean.com'], self.log_dir / 'log_tilt.log', cwd=self.ts_dir)

        rec_file = next((f for f in [
            self.ts_dir / f'{self.ts_name}_rec.mrc',
            self.ts_dir / f'{self.ts_name}.rec'
        ] if f.exists()), None)
        
        if not rec_file:
            self.logger.warning('No reconstruction file found. Cannot perform final rotation.')
            return
            
        self.logger.info(f'Starting final rotation on {rec_file}.')
        rot_file = self.ts_dir / f'{self.ts_name}_rot.mrc'
        run_command(['trimvol', '-rx', str(rec_file), str(rot_file)], self.log_dir / 'log_trimvol.log', cwd=self.ts_dir)
        run_command(['clip', 'flipz', str(rot_file), str(self.final_mrc_path)], self.log_dir / 'log_clip.log', cwd=self.ts_dir)
        rot_file.unlink()
        self.logger.info('Finished final rotation.')

    def _pilot_alignment(self) -> None:
        """Runs the pilot alignment step.
        Back up method not being used right now"""
        self.logger.info('Running pilot alignment...')
        self._prune_fiducial_model([])
        run_command(['submfg', 'align.com'], self.log_dir / 'log_align.log', cwd=self.ts_dir)
        self.logger.info('Pilot alignment completed.')

    def run(self) -> str:
        """Executes the full optimization pipeline for the tomogram."""
        if self.final_mrc_path.exists():
            self.logger.info('Final file already exists. Skipping.')
            return f'Skipped {self.ts_name}: Final file already exists.'

        self.logger.info('Starting optimization.')
        if not self.align_log_path.exists():
            return f"FAILED {self.ts_name}: align.log not found."

        try:
            views_to_exclude, contours_to_include = self._analyze_alignment_logs()

            # if self.ts_name.startswith("20251113_L17_G1"):
                # views_to_exclude.extend([str(i) for i in range(1, 4)])
                # views_to_exclude.extend([str(i) for i in range(21, 30)])

            self.logger.info(f'Pruning: {len(views_to_exclude)} views to exclude and {len(contours_to_include)} contours to include.')
            self._prune_fiducial_model(contours_to_include)
            views_to_exclude_str = ','.join(views_to_exclude) if views_to_exclude else '0'
            self._create_com_files(views_to_exclude_str)

            self.logger.info('Running cleaned alignment...')
            run_command(['submfg', 'align_clean.com'], self.log_dir / 'log_align.log', cwd=self.ts_dir)

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


def optimization_worker(ts_directory: str, tomogram_logs_dir: Path) -> str:
    """
    Worker function for multiprocessing.Pool.
    Instantiates and runs the eTomoOptimizer for a given tomogram directory.
    """
    try:
        optimizer = eTomoOptimizer(ts_directory, tomogram_logs_dir)
        return optimizer.run()
    except Exception as e:
        # Use a generic logger here as this part runs in a separate process
        logging.error(f"Failed to initialize or run optimizer for {ts_directory}: {e}", exc_info=True)
        return f"FAILED {ts_directory}: {e}"


def run_optimization_pipeline(tiltstack_dir: Path, main_logs_dir: Path):
    """Main function to find tomograms and run the optimization in parallel."""
    main_logger = logging.getLogger(__name__)
    
    tomogram_logs_dir = main_logs_dir / 'tomograms'
    tomogram_logs_dir.mkdir(exist_ok=True)
    
    if not tiltstack_dir.is_dir():
        main_logger.error(f"Provided tiltstack directory does not exist: {tiltstack_dir}")
        return

    try:
        tomo_list = sorted([str(p) for p in tiltstack_dir.glob(f'{TOMO_MATCH_STRING}*') if p.is_dir()])
        if not tomo_list:
            main_logger.error(f"No tomogram directories found in {tiltstack_dir} matching pattern: {TOMO_MATCH_STRING}*")
            return
    except Exception as e:
        main_logger.error(f"Error finding tomograms in {tiltstack_dir}: {e}")
        return

    main_logger.info(f"Starting parallel optimization for {len(tomo_list)} tomograms...")
    
    worker_with_logs = partial(optimization_worker, tomogram_logs_dir=tomogram_logs_dir)
    
    with Pool(NUM_CPU_CORES) as pool:
        results = pool.map(worker_with_logs, tomo_list)
        for res in results:
            main_logger.info(res)
            
    main_logger.info("All optimization tasks are complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eTomo optimization script.")
    parser.add_argument(
        '--tiltstack_dir', 
        type=Path, 
        required=True,
        help="Path to the directory containing tomogram subdirectories (e.g., warp_tiltseries/tiltstack)."
    )
    parser.add_argument(
        '--main_logs_dir', 
        type=Path, 
        required=True,
        help="Path to the main logs directory for the project."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    run_optimization_pipeline(args.tiltstack_dir, args.main_logs_dir)
