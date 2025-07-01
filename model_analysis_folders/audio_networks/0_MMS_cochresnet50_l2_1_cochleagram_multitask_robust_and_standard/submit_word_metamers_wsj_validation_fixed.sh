#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_nhm2018_timeavg
#SBATCH --output=output/metamers_%A_%a.out
#SBATCH --error=output/metamers_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-73 # Array indices: 0-36 for robust (sound IDs 0-36), 37-73 for standard (sound IDs 0-36)
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

# Decode the array task ID to get sound ID and model type
# Indices 0-36: robust model (sound IDs 0-36)
# Indices 37-73: standard model (sound IDs 0-36)
SOUND_IDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)

# Task IDs 0-36 = robust, task IDs 37-73 = standard
if [ $SLURM_ARRAY_TASK_ID -le 36 ]; then
    MODEL_TYPE="robust"
    SOUND_IDX=$SLURM_ARRAY_TASK_ID
else
    MODEL_TYPE="standard"
    SOUND_IDX=$((SLURM_ARRAY_TASK_ID - 37))
fi

SOUND_ID=${SOUND_IDS[$SOUND_IDX]}

echo "=== Configuration ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Sound ID: $SOUND_ID"
echo "Model Type: $MODEL_TYPE"
echo "===================="

# Use SLURM job ID as the unique run number (without task ID)
UNIQUE_RUN_NUMBER=$SLURM_ARRAY_JOB_ID
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using SLURM job ID as run number: $UNIQUE_RUN_NUMBER"
echo "METAMER_RUN_NUMBER environment variable: $METAMER_RUN_NUMBER"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Define random seeds to iterate through
RANDOM_SEEDS=(9 400 85)

# Loop through subclips first, then random seeds
for SUBCLIP_IDX in 0 1 2; do
    echo "=== Processing subclip $SUBCLIP_IDX ==="
    for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
        echo "=== Processing Random Seed: $RANDOM_SEED ==="
        echo "=== Processing $MODEL_TYPE model ==="
        
        # Create model-specific output directory with random seed
        OUTPUT_DIR="plots/metamers_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== Directory Setup ==="
        echo "Output Directory: $OUTPUT_DIR"
        echo "Current Directory: $(pwd)"
        echo "======================"

        # Build the command with subclip index, random seed, and our loss function
        CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

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
            continue  # Skip to next iteration if this one fails
        fi

        # --- Plotting section temporarily disabled ---
        # echo "=== Finding Metamer Directory for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
        # # Find the metamer directory for this run
        # METAMER_DIR="metamers_${UNIQUE_RUN_NUMBER}"

        # if [ ! -d "$METAMER_DIR" ]; then
        #     echo "Error: No metamer directory found for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED!"
        #     echo "Current directory contents:"
        #     ls -R
        #     continue  # Skip to next iteration if no directory found
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
        #     --sound_id \"$SOUND_ID\""

        # echo "Plotting command: $PLOT_CMD"
        # echo "============================"

        # eval $PLOT_CMD

        # if [ $? -ne 0 ]; then
        #     echo "Error: Plotting script failed for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED with exit code $?"
        #     continue  # Skip to next iteration if plotting fails
        # fi

        # echo "=== Analysis Complete for $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
        # echo "Results saved in: $OUTPUT_DIR"
        # echo "========================"
    done
done

echo "=== All Processing Complete ===" 