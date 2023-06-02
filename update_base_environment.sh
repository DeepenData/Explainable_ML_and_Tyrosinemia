#!/bin/bash

# Check if requirements.txt file exists
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt file not found."
  exit 1
fi

# Activate the base Conda environment
source activate base

# Install packages from requirements.txt using mamba and pip
while read -r line; do
  if [[ $line == conda* ]]; then
    conda_package=$(echo $line | cut -d '=' -f 1 | cut -d ' ' -f 2)
    mamba install $conda_package --yes
  else
    pip_package=$(echo $line | cut -d '=' -f 1)
    pip install $pip_package
  fi
done < requirements.txt

echo "Base environment updated with packages from requirements.txt"