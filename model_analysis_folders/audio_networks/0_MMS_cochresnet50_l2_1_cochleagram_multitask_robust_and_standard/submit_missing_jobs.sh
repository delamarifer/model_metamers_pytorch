#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_missing_jobs
#SBATCH --output=output/metamers_missing_%A_%a.out
#SBATCH --error=output/metamers_missing_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-11  # Adjust this based on number of missing jobs
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

# Get the run number from the job array ID
RUN_NUMBER=$SLURM_ARRAY_JOB_ID
export METAMER_RUN_NUMBER=$RUN_NUMBER

# Define the missing jobs as arrays
# Format: sound_id, model_type, subclip_idx, random_seed
# Example: If sound 5, standard model, subclip 1, seed 123 is missing:
MISSING_JOBS=(
    "5 standard 1 123"
    "5 robust 1 123"
    "6 standard 1 123"
    "6 robust 1 123"
    "7 standard 1 123"
    "7 robust 1 123"
    "18 standard 1 123"
    "18 robust 1 123"
    "5 standard 2 123"
    "5 robust 2 123"
    "6 standard 2 123"
    "6 robust 2 123"
)

# Get the job parameters for this array task
JOB_PARAMS=(${MISSING_JOBS[$SLURM_ARRAY_TASK_ID]})
SOUND_ID=${JOB_PARAMS[0]}
MODEL_TYPE=${JOB_PARAMS[1]}
SUBCLIP_IDX=${JOB_PARAMS[2]}
RANDOM_SEED=${JOB_PARAMS[3]}

echo "=== Processing Missing Job ==="
echo "Sound ID: $SOUND_ID"
echo "Model Type: $MODEL_TYPE"
echo "Subclip: $SUBCLIP_IDX"
echo "Random Seed: $RANDOM_SEED"
echo "=============================="

# Create output directory
OUTPUT_DIR="plots/metamers_${MODEL_TYPE}_${RUN_NUMBER}_seed${RANDOM_SEED}"
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 0.1 --lr_decay 0.7 -R $RANDOM_SEED"

echo "=== Running Missing Job ==="
echo "Command: $CMD"
echo "=========================="

# Run the metamer generation
$CMD

# Check if metamer generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Metamer generation failed for Sound $SOUND_ID, $MODEL_TYPE, subclip $SUBCLIP_IDX, seed $RANDOM_SEED"
    exit 1
fi

echo "=== Job completed successfully ===" 