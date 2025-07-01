#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=fmri_predictivity_parallel
#SBATCH --output=output/fmri_predictivity_parallel_%A_%a.out
#SBATCH --error=output/fmri_predictivity_parallel_%A_%a.err
#SBATCH --mem=8000
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high
#SBATCH --array=0-13  # 2 models * 7 layers = 14 total jobs (0-13)

set -e
set -o pipefail

# Load CUDA module
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

# Change to the working directory
cd /om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/fmri_predictivity_scripts

# Set SLURM job ID as metamer run number if this script generates metamers
if [ ! -z "$SLURM_JOB_ID" ]; then
    export METAMER_RUN_NUMBER=$SLURM_JOB_ID
    echo "Set METAMER_RUN_NUMBER to SLURM job ID: $SLURM_JOB_ID"
fi

echo "=== Starting fMRI Predictivity Parallel Analysis ==="
echo "Current directory: $(pwd)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "=================================================="

# Define the job mapping
# Array indices 0-6: robust model layers 0-6
# Array indices 7-13: standard model layers 0-6
MODEL_TYPES=("robust" "robust" "robust" "robust" "robust" "robust" "robust" 
             "standard" "standard" "standard" "standard" "standard" "standard" "standard")
LAYER_INDICES=(0 1 2 3 4 5 6 
               0 1 2 3 4 5 6)

# Get the current job's parameters
CURRENT_MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
CURRENT_LAYER_INDEX=${LAYER_INDICES[$SLURM_ARRAY_TASK_ID]}

echo "Processing: Model=$CURRENT_MODEL_TYPE, Layer=$CURRENT_LAYER_INDEX"

# Check if features exist
FEATURE_FILE="./features/natsound_activations_${CURRENT_MODEL_TYPE}.h5"
if [ ! -f "$FEATURE_FILE" ]; then
    echo "ERROR: Feature file not found: $FEATURE_FILE"
    echo "Please run extract_features_only.py first to generate features."
    exit 1
fi

# Run the single layer regression
echo "=== Running regression for $CURRENT_MODEL_TYPE layer $CURRENT_LAYER_INDEX ==="
python run_regression_single_layer.py $CURRENT_MODEL_TYPE $CURRENT_LAYER_INDEX

if [ $? -eq 0 ]; then
    echo "=== Regression completed successfully for $CURRENT_MODEL_TYPE layer $CURRENT_LAYER_INDEX ==="
else
    echo "=== ERROR: Regression failed for $CURRENT_MODEL_TYPE layer $CURRENT_LAYER_INDEX ==="
    exit 1
fi

echo "=== fMRI Predictivity Parallel Analysis Complete ==="
echo "Results saved in: ../regression_results/"
echo "==================================================" 