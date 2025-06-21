#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Change to the correct directory
cd /om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard

# Run the plotting script
python plot_organized_metamers.py ./metamers_organized ./plots_organized 