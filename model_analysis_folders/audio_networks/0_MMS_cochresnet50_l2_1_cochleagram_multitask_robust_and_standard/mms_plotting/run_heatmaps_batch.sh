#!/bin/bash
#SBATCH --job-name=resnet_heatmaps
#SBATCH --output=resnet_heatmaps_%j.out
#SBATCH --error=resnet_heatmaps_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal

# Load modules and activate conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Set working directory
cd /om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch

# Run the heatmap generation script
python model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/mms_plotting/create_metamer_heatmaps.py \
    --pickle_dir model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/mms_plotting/merged_metamers \
    --output_dir model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/mms_plotting/heatmaps

echo "Heatmap generation complete!" 