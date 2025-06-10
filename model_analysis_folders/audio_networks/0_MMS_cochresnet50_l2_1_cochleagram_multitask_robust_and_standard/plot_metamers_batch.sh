#!/bin/bash

# Activate the conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Find the latest metamer directory
METAMERS_DIR="model_analysis_folders/audio_networks/A_MMS_cochresnet50_l2_1_cochleagram_multitask_robust/metamers"
LATEST_DIR=$(ls -td "$METAMERS_DIR"/psychophysics_wsj400_jsintest_inversion_loss_layer_RS* | head -1)
TIMESTAMP=$(basename "$LATEST_DIR" | grep -o '[0-9]\{8\}_[0-9]\{6\}$')

# Set variables
BASE_PATH="$LATEST_DIR"
OUTPUT_FOLDER="model_analysis_folders/audio_networks/A_MMS_cochresnet50_l2_1_cochleagram_multitask_robust/plots_${TIMESTAMP}"
MODEL_TYPE="standard"
LOSS_TYPE="inversion_loss_layer"

echo "Using metamer directory: $BASE_PATH"
echo "Saving plots to: $OUTPUT_FOLDER"

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Edit these arrays as needed
SOUND_IDS=(0)
SEEDS=(0)

for SOUND_ID in "${SOUND_IDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python model_analysis_folders/audio_networks/A_MMS_cochresnet50_l2_1_cochleagram_multitask_robust/make_single_layer_plots_cochmanual_mse.py \
      --base_path "$BASE_PATH" \
      --output_folder "$OUTPUT_FOLDER" \
      --sound_id "$SOUND_ID" \
      --rand_seed_1 "$SEED" \
      --model_type "$MODEL_TYPE" \
      --loss_type "$LOSS_TYPE"
  done
done 