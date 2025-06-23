#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_nhm2018_timeavg
#SBATCH --output=output/metamers_%A_%a.out
#SBATCH --error=output/metamers_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=18,22,23,25 # Process specific sound categories for testing
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high

# Enable error handling
set -e
set -o pipefail

# Load CUDA module if available
module load cuda70/toolkit/7.0.28

source ~/.bashrc
conda activate model_metamers_pytorch

# Get the repository root directory
REPO_ROOT="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch"

# Add the repository root and analysis_scripts to PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/analysis_scripts:$PYTHONPATH

# Create output directory
mkdir -p output

# Print GPU information for debugging
echo "=== GPU Information ==="
nvidia-smi
echo "======================"

echo "=== Configuration ==="
echo "Sound ID: $SLURM_ARRAY_TASK_ID"
echo "===================="

# Use a unique run number based on the current timestamp
UNIQUE_RUN_NUMBER=$(date +%s)
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using unique run number: $UNIQUE_RUN_NUMBER"

# Define random seeds to iterate through
RANDOM_SEEDS=(7 0)

# Loop through subclips first, then random seeds
for SUBCLIP_IDX in 0 1 2; do
    echo "=== Processing subclip $SUBCLIP_IDX ==="
    for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
        echo "=== Processing Random Seed: $RANDOM_SEED ==="
        # Process both model types
        for MODEL_TYPE in "standard" "robust"; do
            echo "=== Processing $MODEL_TYPE model ==="
            
            # Create model-specific output directory with random seed
            OUTPUT_DIR="plots/metamers_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
            mkdir -p "$OUTPUT_DIR"

            echo "=== Directory Setup ==="
            echo "Output Directory: $OUTPUT_DIR"
            echo "Current Directory: $(pwd)"
            echo "======================"

            # Build the command with subclip index, random seed, and our loss function
            CMD="python make_mms.py $SLURM_ARRAY_TASK_ID -I 1 -N 1 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 0.1 --lr_decay 0.7 -R $RANDOM_SEED"
            # CMD="python make_mms.py $SLURM_ARRAY_TASK_ID -I 1 -N 1 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

            echo "=== Running Metamer Generation for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
            echo "Command: $CMD"
            echo "================================="

            # Record start time
            MMS_START_TIME=$(date +%s)
            echo "Start time: $(date -d @$MMS_START_TIME)"

            # Run the metamer generation
            $CMD

            # Record end time and compute elapsed
            MMS_END_TIME=$(date +%s)
            MMS_ELAPSED=$((MMS_END_TIME - MMS_START_TIME))
            echo "End time: $(date -d @$MMS_END_TIME)"
            echo "Elapsed time for MMS generation: ${MMS_ELAPSED} seconds"

            # Check if metamer generation was successful
            if [ $? -ne 0 ]; then
                echo "Error: Metamer generation failed for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED with exit code $?"
                continue  # Skip to next model type if this one fails
            fi

            # --- Plotting section temporarily disabled ---
            # echo "=== Finding Metamer Directory for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
            # # Find the metamer directory for this run
            # METAMER_DIR="metamers_${UNIQUE_RUN_NUMBER}"

            # if [ ! -d "$METAMER_DIR" ]; then
            #     echo "Error: No metamer directory found for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED!"
            #     echo "Current directory contents:"
            #     ls -R
            #     continue  # Skip to next model type if no directory found
            # fi

            # echo "Found metamer directory: $METAMER_DIR"
            # echo "Directory contents:"
            # ls -l "$METAMER_DIR"
            # echo "================================"

            # # Run the plotting script
            # echo "=== Running Plotting Script for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
            # PLOT_CMD="python make_single_layer_plots_cochmanual_mse.py \
            #     --base_path \"$METAMER_DIR\" \
            #     --output_folder \"$OUTPUT_DIR\" \
            #     --rand_seed_1 $RANDOM_SEED \
            #     --model_type \"$MODEL_TYPE\" \
            #     --loss_type \"time_averaged_inversion_loss_layer\" \
            #     --sound_id \"$SLURM_ARRAY_TASK_ID\""

            # echo "Plotting command: $PLOT_CMD"
            # echo "============================"

            # eval $PLOT_CMD

            # if [ $? -ne 0 ]; then
            #     echo "Error: Plotting script failed for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED with exit code $?"
            #     continue  # Skip to next model type if plotting fails
            # fi

            # echo "=== Analysis Complete for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
            # echo "Results saved in: $OUTPUT_DIR"
            # echo "========================"
        done
    done
done

echo "=== All Processing Complete ===" 