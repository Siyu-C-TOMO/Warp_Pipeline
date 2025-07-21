#!/bin/bash

# This script activates the correct conda environment and then runs the appendix pipeline.

# Exit immediately if a command exits with a non-zero status.
set -e

# Source conda activation script.
# Assuming a standard conda installation path. This might need adjustment by the user.
if [ -f "/home/sic027/conda/etc/profile.d/conda.sh" ]; then
    source "/home/sic027/conda/etc/profile.d/conda.sh"
else
    echo "Conda activation script not found. Please adjust the path in this script."
    exit 1
fi

# Activate the specified conda environment
conda activate AlisterEM

# Get the directory where the script is located to find the python script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Run the python script, passing all arguments from this shell script
echo "Running appendix pipeline in conda environment: AlisterEM"
python "$SCRIPT_DIR/run_appendix.py" "$@"
echo "Pipeline script finished."

# Deactivate the environment
conda deactivate
