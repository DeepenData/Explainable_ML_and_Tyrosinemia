#!/bin/bash

# Check if environment name is provided
if [ -z "$1" ]; then
  echo "Please provide the name of the Conda environment."
  exit 1
fi

# Activate the Conda environment
source activate "$1"

# Export Conda packages to conda_packages.txt
conda list --export > conda_packages.txt

# Export pip packages to pip_packages.txt
pip freeze > pip_packages.txt

# Combine Conda and pip package lists
cat conda_packages.txt pip_packages.txt > requirements.txt

# Clean up temporary files
rm conda_packages.txt pip_packages.txt

echo "Environment exported to requirements.txt"
