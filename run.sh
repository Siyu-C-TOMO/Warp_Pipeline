#!/bin/bash

# This script serves as a wrapper to automatically load the required environment
# module before executing the main Python pipeline.

# Get the directory where this script is located to build a robust path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Loading required module: warp/2.0.0dev31"
module load warp/2.0.0dev31

echo "Executing pipeline: clean_pipeline.py"
# Execute the python script, passing along all command-line arguments (e.g., --stage)
python3 "${SCRIPT_DIR}/clean_pipeline.py" "$@"

echo "Pipeline execution finished."
