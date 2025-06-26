#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=fmri_predictivity_analysis
#SBATCH --output=output/fmri_predictivity_analysis_%j.out
#SBATCH --error=output/fmri_predictivity_analysis_%j.err
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high

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

echo "=== Starting fMRI Predictivity Analysis Pipeline ==="
echo "Current directory: $(pwd)"
echo "=================================================="

# Step 1: Run the fMRI predictivity analysis for all layers
echo "=== Step 1: Running fMRI Predictivity Analysis ==="
echo "This will extract features and run regressions for all layers..."
echo "=================================================="

python run_fmri_predictivity_all_layers.py

if [ $? -eq 0 ]; then
    echo "=== fMRI Predictivity Analysis completed successfully ==="
else
    echo "=== ERROR: fMRI Predictivity Analysis failed! ==="
    echo "Check the error logs above for details."
    exit 1
fi

# Step 2: Generate plots from the analysis results
echo "=== Step 2: Generating fMRI Predictivity Plots ==="
echo "This will create plots from the regression results..."
echo "=================================================="

python plot_fmri_predictivity_all_layers.py

if [ $? -eq 0 ]; then
    echo "=== fMRI Predictivity Plotting completed successfully ==="
else
    echo "=== ERROR: fMRI Predictivity Plotting failed! ==="
    echo "Check the error logs above for details."
    exit 1
fi

echo "=== fMRI Predictivity Analysis Pipeline Complete ==="
echo "Results saved in:"
echo "  - Regression results: ../regression_results/"
echo "  - Plots: ./plots_fmri_predictivity_analysis/"
echo "==================================================" 