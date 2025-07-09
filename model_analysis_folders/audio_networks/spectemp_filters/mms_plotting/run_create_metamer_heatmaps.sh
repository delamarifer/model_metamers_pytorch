#!/bin/bash
# Activate your environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Make output and error directories if they don't exist
mkdir -p slurm_out
mkdir -p slurm_err

# Run the heatmap script with unbuffered output
python3 -u create_metamer_heatmaps.py --pickle_dir ../merged_metamers --output_dir ../heatmaps 