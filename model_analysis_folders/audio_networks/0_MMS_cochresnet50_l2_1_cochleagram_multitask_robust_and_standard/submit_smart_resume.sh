#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_smart_resume
#SBATCH --output=output/metamers_smart_%A_%a.out
#SBATCH --error=output/metamers_smart_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=5,6,7,18  # Process specific sound categories for testing
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

# Find the most recent metamers directory
LATEST_RUN=$(ls -d metamers_* 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_RUN" ]; then
    echo "No existing metamers directories found. Starting fresh run."
    RUN_NUMBER=$SLURM_ARRAY_JOB_ID
else
    echo "Found existing run: $LATEST_RUN"
    # Extract run number from directory name
    RUN_NUMBER=$(echo $LATEST_RUN | sed 's/metamers_//')
    echo "Using existing run number: $RUN_NUMBER"
fi

export METAMER_RUN_NUMBER=$RUN_NUMBER

# Define parameters
RANDOM_SEEDS=(42 123 456)
MODEL_TYPES=("standard" "robust")
SUBCLIPS=(0 1 2)

# Function to check if a specific combination already exists
check_if_exists() {
    local sound_id=$1
    local model_type=$2
    local subclip_idx=$3
    local random_seed=$4
    
    local expected_pickle_path="metamers_${RUN_NUMBER}/${sound_id}_SOUND_word_${model_type^^}_subclip${subclip_idx}/all_metamers_pickle.pckl"
    
    if [ -f "$expected_pickle_path" ]; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to run a single metamer generation
run_single_metamer() {
    local sound_id=$1
    local model_type=$2
    local subclip_idx=$3
    local random_seed=$4
    
    echo "=== Running: Sound $sound_id, $model_type, Subclip $subclip_idx, Seed $random_seed ==="
    
    # Create output directory
    OUTPUT_DIR="plots/metamers_${model_type}_${RUN_NUMBER}_seed${random_seed}"
    mkdir -p "$OUTPUT_DIR"
    
    # Build the command
    CMD="python make_mms.py $sound_id -I 3000 -N 8 -M $model_type -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $subclip_idx -Z 0.1 --lr_decay 0.7 -R $random_seed"
    
    echo "Command: $CMD"
    
    # Run the metamer generation
    $CMD
    
    if [ $? -ne 0 ]; then
        echo "Error: Metamer generation failed for Sound $sound_id, $model_type, subclip $subclip_idx, seed $random_seed"
        return 1
    fi
    
    echo "=== Successfully completed: Sound $sound_id, $model_type, Subclip $subclip_idx, Seed $random_seed ==="
    return 0
}

# Main processing loop
echo "=== Starting Smart Resume Processing ==="
echo "Run Number: $RUN_NUMBER"
echo "Sound ID: $SLURM_ARRAY_TASK_ID"
echo "======================================"

TOTAL_JOBS=0
SKIPPED_JOBS=0
COMPLETED_JOBS=0
FAILED_JOBS=0

for RANDOM_SEED in "${RANDOM_SEEDS[@]}"; do
    for SUBCLIP_IDX in "${SUBCLIPS[@]}"; do
        for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            if check_if_exists $SLURM_ARRAY_TASK_ID $MODEL_TYPE $SUBCLIP_IDX $RANDOM_SEED; then
                echo "✓ SKIPPING: Sound $SLURM_ARRAY_TASK_ID, $MODEL_TYPE, Subclip $SUBCLIP_IDX, Seed $RANDOM_SEED (already exists)"
                SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            else
                echo "→ RUNNING: Sound $SLURM_ARRAY_TASK_ID, $MODEL_TYPE, Subclip $SUBCLIP_IDX, Seed $RANDOM_SEED"
                
                if run_single_metamer $SLURM_ARRAY_TASK_ID $MODEL_TYPE $SUBCLIP_IDX $RANDOM_SEED; then
                    COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                else
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                fi
            fi
        done
    done
done

echo ""
echo "=== SUMMARY ==="
echo "Total jobs: $TOTAL_JOBS"
echo "Skipped (already exist): $SKIPPED_JOBS"
echo "Completed: $COMPLETED_JOBS"
echo "Failed: $FAILED_JOBS"
echo "=================="

if [ $FAILED_JOBS -gt 0 ]; then
    echo "WARNING: $FAILED_JOBS jobs failed!"
    exit 1
else
    echo "All jobs completed successfully!"
fi 