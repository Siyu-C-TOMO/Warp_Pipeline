#!/usr/bin/env python3

import logging
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Optional

# --- Data Reorganization ---

def reorganize_falcon4_data(config, logs_dir: Path):
    """
    Moves and organizes raw Falcon4 data from a source directory to the processing directory.

    This function is designed to be run when `camera_type` is 'Falcon4'. It moves
    files from a temporary source location (`falcon4_source_dir`) to the final
    data directory (`raw_directory`/`dataset_name`). It uses batch `mv` commands
    for performance.

    The logic is as follows:
    1. A 'frames' directory is created in the destination.
    2. Files ending in .eer, .eer.mdoc, and the gain reference file are moved into
       the 'frames' directory.
    3. All other files and directories (e.g., 'mdocs' folder, 'nav.nav', 'atlas', etc.)
       are moved to the root of the destination directory.

    Args:
        config: The configuration module object (e.g., config.py).
        logs_dir: The path to the main logging directory for the run.
    """
    source_dir = Path(config.falcon4_source_dir)
    dest_dir = Path(config.raw_directory) / config.dataset_name
    reorg_log_path = logs_dir / "reorg.log"

    if not source_dir.is_dir() or not any(source_dir.glob('*.eer')):
        logging.info(f"Source directory '{source_dir}' has no .eer files or does not exist. Skipping reorganization.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = dest_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    logging.info("Starting Falcon4 data reorganization (optimized)...")

    gain_ref_name = Path(config.gain_ref).name
    
    to_frames = []
    to_root_files = []
    to_root_dirs = []
    
    counts = {
        'eer': 0,
        'mdoc': 0,
        'gain': 0,
        'other_files': 0,
        'dirs': 0
    }

    for item_path in list(source_dir.iterdir()):
        if item_path.is_file():
            if item_path.name.endswith('.eer'):
                to_frames.append(str(item_path))
                counts['eer'] += 1
            elif item_path.name.endswith('.eer.mdoc'):
                to_frames.append(str(item_path))
                counts['mdoc'] += 1
            elif item_path.name == gain_ref_name:
                to_frames.append(str(item_path))
                counts['gain'] += 1
            else:
                to_root_files.append(str(item_path))
                counts['other_files'] += 1
        elif item_path.is_dir():
            to_root_dirs.append(str(item_path))
            counts['dirs'] += 1

    if to_frames:
        cmd = ['mv', '-t', str(frames_dir)] + to_frames
        run_command(cmd, reorg_log_path, verbose=False)
    
    if to_root_files:
        cmd = ['mv', '-t', str(dest_dir)] + to_root_files
        run_command(cmd, reorg_log_path, verbose=False)

    if to_root_dirs:
        cmd = ['mv', '-t', str(dest_dir)] + to_root_dirs
        run_command(cmd, reorg_log_path, verbose=False)
        
    logging.info("Reorganization Summary:")
    logging.info(f"  - Moved {counts['eer']} .eer files to frames/")
    logging.info(f"  - Moved {counts['mdoc']} .eer.mdoc files to frames/")
    if counts['gain'] > 0:
        logging.info(f"  - Moved {counts['gain']} gain reference file(s) to frames/")
    logging.info(f"  - Moved {counts['other_files']} other files and {counts['dirs']} directories to the destination root.")
    logging.info("Falcon4 data reorganization completed.")


# --- Custom Exceptions ---

class LogParsingError(Exception):
    """Custom exception for errors during log file parsing."""
    pass

# --- Command Execution ---

def run_command(
    command: List[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    verbose: bool = True,
) -> None:
    """
    Runs a command, logs its output, and handles errors.

    Args:
        command: The command to run as a list of strings.
        log_path: Path to the log file for stdout and stderr.
        cwd: The working directory for the command. Defaults to None.
        env: Environment variables for the command. Defaults to None.
        shell: Whether to use the shell. Defaults to False.
        verbose: If True, prints command info to the main logger.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        logging.info(f"Running command: {' '.join(command)}")
        if cwd:
            logging.info(f"Working directory: {cwd}")

    try:
        with open(log_path, 'w') as log_file:
            subprocess.run(
                command,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                shell=shell
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}.")
        logging.error(f"Check the log for details: {log_path.resolve()}")
        raise
    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}. Ensure it is in the system's PATH.")
        raise

# --- XML Parsing (from xml_parser.py) ---

xml_logger = logging.getLogger("xml_updater")

def _get_excludelist_from_com(align_com_path: Path) -> np.ndarray:
    """Reads an eTomo .com file and returns the ExcludeList as a numpy array."""
    if not align_com_path.exists():
        xml_logger.warning(f"{align_com_path} not found, skipping exclusion list.")
        return np.array([])

    excludelist_str = None
    with align_com_path.open('r') as f:
        for line in f:
            if line.strip().startswith('ExcludeList'):
                # Assumes format is "ExcludeList   1,2,3"
                excludelist_str = line.split(maxsplit=1)[1].strip()
                break
    
    if excludelist_str:
        try:
            # Filter out any empty strings that might result from splitting
            items = [int(i) for i in excludelist_str.split(',') if i]
            return np.array(items)
        except (ValueError, IndexError) as e:
            xml_logger.error(f"Could not parse ExcludeList '{excludelist_str}': {e}")
            return np.array([])
    
    return np.array([])

def _edit_mrcxml_usetilts(xml_path: Path, excludelist: np.ndarray):
    """Parses an MRC-XML file and deactivates tilts based on the excludelist."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    use_tilt_element = root.find('UseTilt')

    if use_tilt_element is None or use_tilt_element.text is None:
        xml_logger.warning(f"No 'UseTilt' section found in {xml_path}. Skipping.")
        return

    if excludelist.size > 0:
        xml_logger.info(f"Found {len(excludelist)} tilts to exclude in {xml_path.name}.")
        
        # Account for 0-based indexing vs 1-based in eTomo
        exclude_indices = excludelist - 1
        
        use_tilts_list = use_tilt_element.text.strip().split()
        
        for i in exclude_indices:
            if 0 <= i < len(use_tilts_list):
                use_tilts_list[i] = 'False'
            else:
                xml_logger.warning(f"Index {i} is out of bounds for UseTilt list in {xml_path.name}")

        # Reconstruct the string without extra leading/trailing newlines
        use_tilt_element.text = '\n'.join(use_tilts_list)
        
        # This attribute update might be redundant depending on the parser, but let's be safe
        root.set('UseTilt', use_tilt_element.text)
        
        tree.write(xml_path, xml_declaration=True, encoding='UTF-8')
        xml_logger.info(f"Successfully modified and saved {xml_path.name}.")
    else:
        xml_logger.info(f"No tilts to exclude for {xml_path.name}. File is unchanged.")

def update_xml_files_from_com(base_dir: Path):
    """
    Main function to parse and edit all MRC-XML files in a directory
    based on their corresponding eTomo 'align_clean.com' files.
    """
    xml_logger.info("Starting XML parsing and updating process...")
    
    xml_dir = base_dir
    tiltstack_dir = base_dir / "tiltstack"

    if not xml_dir.is_dir() or not tiltstack_dir.is_dir():
        xml_logger.error(f"Required directories not found: {xml_dir} or {tiltstack_dir}")
        return

    for xml_file in sorted(xml_dir.glob('*.xml')):
        ts_base_name = xml_file.stem
        xml_logger.info(f"Processing {ts_base_name}...")
        
        align_com_path = tiltstack_dir / ts_base_name / 'align_clean.com'
        
        excludelist = _get_excludelist_from_com(align_com_path)
        _edit_mrcxml_usetilts(xml_file, excludelist)
        
    xml_logger.info("XML update process completed.")


# --- Log and Fiducial Parsing (from etomo_optimize.py) ---

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
        tokens = line.split()
        data_lines.append(tokens[:len(header)])
        
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
    bad_point_df = parse_section(iter(lines), ['obj', 'cont', 'view'])

    return view_df, contour_df, bad_point_df

def read_fiducial_file(path_to_fiducial_file: Path) -> pd.DataFrame:
    """Reads a fiducial point file into a pandas DataFrame."""
    return pd.read_csv(
        path_to_fiducial_file,
        sep=r'\s+',
        header=None,
        names=['object', 'contour', 'x', 'y', 'z'],
        engine='python'
    )
