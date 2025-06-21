#!/bin/bash

# Activate your conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Run the reorganization script
python3 reorganize_metamers.py --base-dir . 