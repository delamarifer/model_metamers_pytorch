#!/bin/bash

# Activate your conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Run the reorganization script (dry run first)
python reorganize_metamers.py --base-dir . --dry-run

# Uncomment the next line to actually move files after checking the dry run
# python reorganize_metamers.py --base-dir . 