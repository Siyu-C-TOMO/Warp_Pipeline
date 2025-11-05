#!/bin/bash

# This script serves as a wrapper to automatically load the required environment
# module before executing the main Python pipeline.

# Get the directory where this script is located to build a robust path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Loading required module: aretomo and warp/2"
module load aretomo
module load warp/2.0.0dev36

echo "Executing pipeline: run_pipeline.py"
# Execute the python script, passing along all command-line arguments (e.g., --stage)
python3 "${SCRIPT_DIR}/run_pipeline.py" "$@"

echo "Pipeline execution finished."
