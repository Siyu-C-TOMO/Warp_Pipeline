#!/usr/bin/env python3

import logging
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterator, Dict, Optional, Union

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
            elif item_path.name.endswith('.gain'):
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
    logging.info(f"  - Moved {counts['gain']} gain reference file(s) to frames/")
    logging.info(f"  - Moved {counts['other_files']} other files and {counts['dirs']} directories to the destination root.")
    logging.info("Falcon4 data reorganization completed.")


def prepare_gain_reference(config, frame_source_path: Path, frames_dir: Path) -> Optional[Path]:
    """
    Finds the correct gain reference file and creates a symlink in the frames directory.
    
    It first checks for the gain file specified in the config. If not found,
    it searches for any .gain file in the source directory.
    """
    gain_ref_path = None
    
    specific_gain_path = frame_source_path / config.gain_ref
    if specific_gain_path.exists():
        gain_ref_path = specific_gain_path
        logging.info(f"Using specified gain file: {gain_ref_path}")
    else:
        logging.warning(f"{config.gain_ref} not found. Searching for other .gain files...")
        found_gains = list(frame_source_path.glob("*.gain"))
        if len(found_gains) == 1:
            gain_ref_path = found_gains[0]
            logging.info(f"Found and using alternative gain file: {gain_ref_path}")
        elif len(found_gains) > 1:
            logging.error(f"Multiple .gain files found in {frame_source_path}. Please specify the correct one in config.py.")
            return
        else:
            logging.error(f"No .gain files found in {frame_source_path}.")
            return

    if gain_ref_path:
        gain_ref_link = frames_dir / gain_ref_path.name
        if not gain_ref_link.exists():
            gain_ref_link.symlink_to(gain_ref_path)
            logging.info(f"Created symlink for gain reference: {gain_ref_link}")
        return gain_ref_link
    
    return None


# --- Custom Exceptions ---

class LogParsingError(Exception):
    """Custom exception for errors during log file parsing."""
    pass

# --- Command Execution ---

def run_command(
    command: Union[List[str], str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    verbose: bool = True,
    module_load: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Runs a command, logs its output, and handles errors.

    Args:
        command: The command to run as a list or a single string.
        log_path: Path to the log file for stdout and stderr.
        cwd: The working directory for the command. Defaults to None.
        env: Environment variables for the command. Defaults to None.
        shell: Whether to use the shell. For complex commands or module loading,
               this will be forced to True.
        verbose: If True, prints command info to the main logger.
        module_load: A module or list of modules to load before running the command.
                     e.g., 'cryolo' or ['module1', 'module2'].
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    use_shell = shell or bool(module_load)
    if use_shell:
        cmd_str = ' '.join(map(str, command)) if isinstance(command, list) else command
        if module_load:
            modules = [module_load] if isinstance(module_load, str) else module_load
            load_prefix = '; '.join(f"module load {mod}" for mod in modules)
            executable_command = f"{load_prefix}; {cmd_str}"
        else:
            executable_command = cmd_str
    else:
        executable_command = command

    if verbose:
        log_cmd_str = executable_command if isinstance(executable_command, str) else ' '.join(map(str, executable_command))
        logging.info(f"Running command: {log_cmd_str}")
        if cwd:
            logging.info(f"Working directory: {cwd}")

    try:
        with open(log_path, 'a') as log_file:
            subprocess.run(
                executable_command,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                shell=use_shell,
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}.")
        logging.error(f"Check the log for details: {log_path.resolve()}")
        raise
    except FileNotFoundError:
        cmd_name = command[0] if isinstance(command, list) else command.split()[0]
        logging.error(f"Command not found: {cmd_name}. Ensure it is in the system's PATH.")
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

# --- Patch Size Calculation ---

def calculate_patch_size(config) -> int:
    """
    Calculates the optimal patch size for eTomo based on configuration.

    If `use_dynamic_patch_size` is False, it returns `default_patch_size`.
    Otherwise, it computes a target size in Angstroms based on the image's 
    longest dimension, pixel size, and a division factor, then finds the 
    closest match from a list of predefined possible sizes.

    Args:
        config: The configuration module object.

    Returns:
        The calculated patch size as an integer.
    """
    if not config.use_dynamic_patch_size:
        logging.info(f"Dynamic patch size disabled. Using default: {config.default_patch_size}")
        return config.default_patch_size

    image_x, image_y = config.pipeline_params['original_x_y_size']
    pixel_size = config.angpix
    longest_dim_px = max(image_x, image_y)
    longest_dim_angstrom = longest_dim_px * pixel_size
    
    target_size = longest_dim_angstrom / config.patch_size_division_factor
    
    possible_sizes = np.array(config.possible_patch_sizes)
    closest_size_index = np.abs(possible_sizes - target_size).argmin()
    final_patch_size = possible_sizes[closest_size_index]
    
    logging.info("Dynamic patch size enabled.")
    logging.info(f"  - Longest image dimension: {longest_dim_px}px")
    logging.info(f"  - Pixel size: {pixel_size} Å/px")
    logging.info(f"  - Longest image dimension in Å: {longest_dim_angstrom:.2f} Å")
    logging.info(f"  - Target patch size (~1/{config.patch_size_division_factor}): {target_size:.2f} Å")
    logging.info(f"  - Selected patch size from list {config.possible_patch_sizes}: {final_patch_size}")
    
    return int(final_patch_size)
