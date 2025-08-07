# Warp & eTomo Cryo-ET Pipeline

A robust and flexible pipeline designed to streamline cryo-electron tomography (cryo-ET) data processing by integrating the capabilities of **Warp** and **IMOD/eTomo**. This pipeline automates the workflow from raw data import to final tomogram reconstruction and optimization.

## Table of Contents

- [About The Project](#about-the-project)
  - [Key Features](#key-features)
  - [Workflow Diagram](#workflow-diagram)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Run](#how-to-run)
  - [1. Configure the Pipeline](#1-configure-the-pipeline)
  - [2. Execute the Wrapper Script](#2-execute-the-wrapper-script)
- [Configuration Guide](#configuration-guide)
  - [General Settings](#general-settings)
  - [Key Acquisition Parameters](#key-acquisition-parameters)
  - [Camera-Specific Settings](#camera-specific-settings)
  - [Computing Resources](#computing-resources)
  - [Derived Parameters (Do Not Edit)](#derived-parameters-do-not-edit)
- [Pipeline Stages Explained](#pipeline-stages-explained)
  - [Stage 1: `preprocess`](#stage-1-preprocess)
  - [Stage 2: `etomo`](#stage-2-etomo)
  - [Stage 3: `optimize`](#stage-3-optimize)
  - [Stage 4: `postprocess`](#stage-4-postprocess)
- [Logging System](#logging-system)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About The Project

This project provides a powerful, automated pipeline for cryo-ET data processing. It handles the entire workflow, from organizing raw data to optimizing tilt-series alignment with a custom eTomo routine. The code has been carefully refactored to be modular, robust, and easy to maintain.

### Key Features

*   **Multi-Camera Support**: Seamlessly processes data from both K3 (.tif) and Falcon4 (.eer) cameras.
*   **Automated Workflow**: Integrates Warp and a custom eTomo optimization routine to streamline processing.
*   **Centralized Configuration**: Manages all parameters in a single `config.py` file.
*   **Advanced eTomo Optimization**: A parallelized stage identifies and excludes high-error views and fiducials to improve alignment.
*   **Flexible Gain Handling**: Automatically finds the gain reference if not explicitly configured.
*   **Structured Logging**: Generates organized logs for easy tracking and debugging.

### Workflow Diagram

The pipeline is organized into four distinct stages that can be run sequentially or individually.

```
+----------------------+
|   Input: Raw Data    |
| (.tif/.eer, .mdoc)   |
+----------------------+
           |
           v
+----------------------+      [1. Prepares data, runs motion correction & CTF]
|   Stage: preprocess  |
+----------------------+
           |
           v
+----------------------+      [2. Runs initial eTomo alignment via Warp]
|     Stage: etomo     |
+----------------------+
           |
           v
+----------------------+      [3. Custom eTomo optimization loop]
|    Stage: optimize   |-----> - Analyzes align.log for outliers
+----------------------+      - Prunes fiducial model
           |                  - Re-runs alignment and reconstruction
           v
+----------------------+      [4. Imports final alignment, performs deconvolution]
|  Stage: postprocess  |
+----------------------+
           |
           v
+----------------------+
|  Output: Optimized   |
|      Tomograms       |
+----------------------+
```

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

*   A Linux environment with support for `module` commands.
*   **Python 3.x**
*   **IMOD**: [IMOD Website](https://bio3d.colorado.edu/imod/)
*   **Warp**: [Warp Website](https://www.warpem.com/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Siyu-C-TOMO/Warp_Pipeline.git
    ```
2.  **Navigate into the directory:**
    ```sh
    cd Warp_Pipeline
    ```
3.  **Install required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Make the wrapper script executable** (only needs to be done once):
    ```sh
    chmod +x warp_wrapper.sh
    ```

## How to Run

The main entry point is the `warp_wrapper.sh` script. It ensures the correct environment is loaded before executing the Python pipeline.

### 1. Configure the Pipeline

Edit the parameters in `config.py` to match your dataset and processing requirements.

### 2. Execute the Wrapper Script

Run the script from the directory where you want the processing folders and logs to be created.

```sh
# Location for processing output
/path/to/your/processing_project/

# Execute the script from there
/path/to/Warp_Pipeline/warp_wrapper.sh --stage all
```

**Command-Line Arguments:**

*   `--stage`: Specifies which part of the pipeline to run.
    *   `all`: (Default) Runs the full pipeline from start to finish.
    *   `preprocess`: Runs only the preprocessing stage.
    *   `etomo`: Runs only the initial eTomo alignment stage.
    *   `optimize`: Runs only the custom eTomo optimization stage.
    *   `postprocess`: Runs only the final post-processing stage.

**Examples:**

```sh
# Run the full pipeline
./path/to/Warp_Pipeline/warp_wrapper.sh --stage all

# Run only the custom optimization and post-processing
./path/to/Warp_Pipeline/warp_wrapper.sh --stage optimize
./path/to/Warp_Pipeline/warp_wrapper.sh --stage postprocess
```

## Configuration Guide

All user-editable settings are located at the top of `config.py`.

### General Settings

*   `dataset_name`: A unique name for your processing run.
    *   **For K3 data**, this should match the name of the raw data folder.
    *   **For Falcon4 data**, you can define a new name for the reorganized dataset.
*   `raw_directory`: The root path where your raw data is, or will be, stored.
    *   **For K3 data**, this is the directory containing the `dataset_name` folder.
    *   **For Falcon4 data**, this is the target directory where the reorganized data will be moved.
*   `Data Structure & Identifiers`:
    *   `frame_folder`: Subfolder for raw frames (e.g., `"frames"`).
    *   `mdoc_folder`: Subfolder for `.mdoc` files (e.g., `"mdocs"`).
*   `gain_ref`: Filename of the gain reference. If the specified file isn't found, the pipeline will automatically use any other `.gain` file present in the `frame_folder`.
*   `tomo_match_string`: A string to identify tomograms by filename (e.g., `"sq"`).

### Key Acquisition Parameters

*   `angpix`: Angstroms per pixel.
*   `dose`: Total exposure dose per tilt for the tilt-series.
*   `tilt_axis_angle`: The rotation angle of the tilt-axis.
*   `thickness_pxl`: Estimated tomogram thickness in pixels.
*   `camera_type`: Switch between `"K3"` or `"Falcon4"`.

### Camera-Specific Settings

*   **For Falcon4:**
    *   `falcon4_source_dir`: The source directory of your raw Falcon4 data.
        > **:warning: CRITICAL:** The contents of this directory will be **MOVED**, not copied, to the `raw_directory` location for processing. **NO BACKUP IS KEPT** in the original location. It is strongly not suggested to make any further changes in the `raw_directory` folder because the raw data is saved and re-organized there.
    *   `falcon4_eer_ngroups`: The number of groups to split each `.eer` file into for motion correction.
*   **For K3:**
    *   `k3_frame_num`: The number of frames per tilt in your K3 data.

### Computing Resources

*   `gpu_devices`: A list of GPU device IDs to use (e.g., `[0, 1]`).
*   `jobs_per_gpu`: Number of parallel jobs to run on each specified GPU.
*   `etomo_cpu_cores`: Number of CPU cores to use for the parallel eTomo optimization stage.

### Derived Parameters (Do Not Edit)

The section below the user settings in `config.py` contains parameters that are automatically calculated based on your inputs. **Do not modify this section unless you completely understand what you are doing.**

## Pipeline Stages Explained

### Stage 1: `preprocess`

**Goal**: Prepare data for alignment.
*   **Key Steps**:
    1.  If `camera_type` is "Falcon4", it first runs `reorganize_falcon4_data` to move and structure the data.
    2.  Creates symbolic links to raw frames, `.mdoc` files, and the gain reference in the processing directory.
    3.  Generates `warp_frameseries.settings` and `warp_tiltseries.settings` using `WarpTools create_settings`.
    4.  Runs frame-level motion correction and CTF estimation (`WarpTools fs_motion_and_ctf`).
    5.  Imports tilt-series metadata from `.mdoc` files to create the initial `tomostar` files (`WarpTools ts_import`).

### Stage 2: `etomo`

**Goal**: Perform initial tilt-series alignment using eTomo within Warp.
*   **Key Steps**:
    1.  Calculates the optimal patch size for alignment, either dynamically or using a fixed default.
    2.  Runs `WarpTools ts_etomo_patches` to perform alignment. This step uses custom IMOD wrappers to ensure compatibility.

### Stage 3: `optimize`

**Goal**: Refine the eTomo alignment by removing high-error views and fiducials.
*   **Key Steps**:
    1.  The `etomo_optimize.py` script is executed in parallel across all tomograms.
    2.  For each tomogram, it parses the `align.log` to identify views and fiducial contours with high residual errors (based on standard deviation).
    3.  It creates a new, pruned fiducial model (`.fid` file) that excludes the identified outliers.
    4.  It generates cleaned `.com` files (`align_clean.com`, `tilt_clean.com`) with an `ExcludeList` parameter.
    5.  It re-runs the alignment (`submfg align_clean.com`) and tomogram reconstruction (`submfg tilt_clean.com`).

### Stage 4: `postprocess`

**Goal**: Finalize the tomograms.
*   **Key Steps**:
    1.  Imports the newly optimized alignments from the `optimize` stage back into Warp (`WarpTools ts_import_alignments`).
    2.  Checks and corrects for defocus handedness issues.
    3.  Estimates the tilt-series CTF (`WarpTools ts_ctf`).
    4.  Parses the final `align_clean.com` files to update Warp's `.xml` files, ensuring bad tilts are excluded from the final reconstruction.
    5.  Performs deconvolution on the final reconstructed tomograms using IMOD's `reducefiltvol`.

## Logging System

The pipeline generates a structured set of logs inside your processing directory (`dataset_name`).

*   `logs/pipeline.log`: The main log file for the entire pipeline run.
*   `logs/<stage_name>.log`: A summary log for each major stage (e.g., `etomo_optimization.log`, `xml_parsing.log`).
*   `logs/tomograms/<tomo_name>/`: A dedicated directory for each tomogram, containing detailed logs for specific processing steps, most notably the `optimization.log` from the custom refinement stage.

## License

Distributed under the MIT License.

## Contact

Siyu Chen - sic027@ucsd.edu

Project Link: [https://github.com/Siyu-C-TOMO/Warp_Pipeline](https://github.com/Siyu-C-TOMO/Warp_Pipeline)

## Acknowledgements

*   Thanks to Joshua Hutchings for the original idea and initial code implementation for the alignment optimization part of this pipeline.
