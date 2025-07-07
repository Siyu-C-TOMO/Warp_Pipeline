# Warp Pipeline

A collection of scripts to streamline cryo-ET data processing workflows, integrating Warp and IMOD/eTomo.

## About The Project

This project provides a flexible, robust pipeline for cryo-ET data processing. It automates several stages, including preprocessing, eTomo-based alignment optimization, and post-processing.

The pipeline has been refactored for clarity, robustness, and ease of maintenance.

### Key Features

*   **Multi-Camera Support**: Seamlessly processes data from both **K3 (.tif)** and **Falcon4 (.eer)** cameras.
*   **Centralized Configuration**: A single `config.py` file manages all parameters. Switching between camera types is as simple as changing one variable.
*   **Simplified Execution**: A wrapper script handles environment module loading automatically.
*   **Modular Architecture**: The code is organized into a main orchestrator, utility modules, and stage-specific scripts.
*   **Automated Workflow**: Automates data linking and settings creation for Warp.
*   **Advanced Optimization**: Includes a parallelized eTomo optimization stage for improved alignment.
*   **Structured Logging**: A comprehensive and organized logging system for easy tracking and debugging.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   A shell environment with support for `module` commands.
*   Python 3
*   IMOD
*   Warp

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/Siyu-C-TOMO/Warp_Pipeline.git
    ```
2.  Navigate into the directory
    ```sh
    cd Warp_Pipeline
    ```
3.  Install Python packages
    ```sh
    pip install -r requirements.txt
    ```
4.  Make the run script executable (only needs to be done once)
    ```sh
    chmod +x warp_wrapper.sh
    ```

## Usage

The main entry point for the pipeline is now the `warp_wrapper.sh` wrapper script. It automatically handles loading the necessary `warp` module before executing the pipeline.

1.  **Configure the pipeline**: Edit the parameters in `config.py` to match your dataset and processing requirements. The most important setting is `camera_type`.

    *   Set `camera_type = "K3"` for K3 data (`.tif` files).
    *   Set `camera_type = "Falcon4"` for Falcon4 data (`.eer` files).

    The script will automatically use the correct parameters based on this setting.

2.  **Run the pipeline**: Execute the `run.sh` script from the `Warp_Pipeline` directory. You can run all stages or specific ones using the `--stage` argument.

    ```sh
    # Run the full pipeline from start to finish
    ./warp_wrapper.sh --stage all

    # Run only the post-processing stage
    ./warp_wrapper.sh --stage postprocess
    ```

## Logging System

The pipeline generates a structured set of logs inside your processing directory (`dataset_name` defined in `config.py`).

*   `logs/pipeline.log`: The main log file for the entire pipeline run.
*   `logs/<stage_name>.log`: A summary log for each major stage (e.g., `etomo_optimization.log`, `xml_parsing.log`).
*   `logs/tomograms/<tomo_name>/`: A dedicated directory for each tomogram, containing detailed logs for specific processing steps like optimization and deconvolution.

## License

Distributed under the MIT License.

## Contact

Siyu Chen - sic027@ucsd.edu

Project Link: [https://github.com/Siyu-C-TOMO/Warp_Pipeline](https://github.com/Siyu-C-TOMO/Warp_Pipeline)

## Acknowledgements

*   Thanks to Joshua Hutchings for the original idea and initial code implementation for the alignment optimization part of this pipeline.
