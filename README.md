# Warp Pipeline

A collection of scripts to streamline cryo-ET data processing workflows, integrating tools like Warp and IMOD/eTomo.

## About The Project

This project provides a flexible pipeline for cryo-ET data processing. It automates several stages, including preprocessing, eTomo-based alignment optimization, and post-processing.

Key features:
*   Automated data linking and settings creation for Warp.
*   Parallelized eTomo optimization for improved alignment.
*   XML parsing to exclude bad tilts from reconstructions.
*   Structured logging for clear and traceable execution.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3
*   IMOD
*   Warp

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/Siyu-C-TOMO/Warp_Pipeline.git
    ```
2.  Install Python packages
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The main entry point for the pipeline is `clean_pipeline.py`.

1.  **Configure the pipeline**: Edit the parameters in `config.py` to match your dataset and processing requirements.

2.  **Run the pipeline**: Execute the main script from your project's parent directory. You can run all stages or specific ones.

    ```sh
    # Run the full pipeline
    python /path/to/Warp_Pipeline/clean_pipeline.py --stage all

    # Run only the eTomo optimization stage
    python /path/to/Warp_Pipeline/clean_pipeline.py --stage etomo
    ```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Siyu Chen - sic027@ucsd.edu

Project Link: [https://github.com/Siyu-C-TOMO/Warp_Pipeline](https://github.com/Siyu-C-TOMO/Warp_Pipeline)

## Acknowledgements

*   Thanks to Joshua Hutchings for the original idea and initial code implementation for the alignment optimization part of this pipeline.
