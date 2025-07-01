#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_missing_rerun
#SBATCH --output=output/rerun_missing_%A_%a.out
#SBATCH --error=output/rerun_missing_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
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

# Define the combinations to run (model_type, random_seed, sound_ids, subclip_indices)
COMBINATIONS=(
    "standard 85 5 6 22 23 24 25 26 27 28 29 30 31 33 34 35 36 0 1 2"
    "standard 9 5 6 22 23 24 25 26 27 28 29 30 31 33 36 0 1 2"
    "standard 400 5 6 22 23 24 25 26 27 28 29 30 31 33 34 35 36 0 1 2"
    "robust 85 2 14 15 36 0 1 2"
    "robust 400 2 14 15 36 0 1 2"
    "robust 9 2 14 15 36 0 1 2"
)

# Get the combination for this array task
COMBINATION_STR="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL_TYPE RANDOM_SEED SOUND_IDS_STR SUBCLIP_INDICES_STR <<< "$COMBINATION_STR"

# Convert space-separated strings to arrays
IFS=' ' read -ra SOUND_IDS <<< "$SOUND_IDS_STR"
IFS=' ' read -ra SUBCLIP_INDICES <<< "$SUBCLIP_INDICES_STR"

echo "=== Configuration ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Type: $MODEL_TYPE"
echo "Random Seed: $RANDOM_SEED"
echo "Sound IDs: ${SOUND_IDS[*]}"
echo "Subclip Indices: ${SUBCLIP_INDICES[*]}"
echo "===================="

# Use a new unique run number for the rerun
UNIQUE_RUN_NUMBER=$((SLURM_ARRAY_JOB_ID + 2000000))  # Offset to avoid conflicts
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using new run number: $UNIQUE_RUN_NUMBER"
echo "METAMER_RUN_NUMBER environment variable: $METAMER_RUN_NUMBER"

# Loop through sound IDs and subclip indices
for SOUND_ID in "${SOUND_IDS[@]}"; do
    for SUBCLIP_IDX in "${SUBCLIP_INDICES[@]}"; do
        echo "=== Processing sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        
        # Create model-specific output directory
        OUTPUT_DIR="plots/rerun_missing_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== Directory Setup ==="
        echo "Output Directory: $OUTPUT_DIR"
        echo "Current Directory: $(pwd)"
        echo "======================"

        # Build the command
        CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

        echo "=== Running Command ==="
        echo "$CMD"
        echo "======================"

        # Run the command
        eval $CMD

        echo "=== Completed sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        echo ""
    done
done

echo "=== All combinations completed for array task $SLURM_ARRAY_TASK_ID ==="
