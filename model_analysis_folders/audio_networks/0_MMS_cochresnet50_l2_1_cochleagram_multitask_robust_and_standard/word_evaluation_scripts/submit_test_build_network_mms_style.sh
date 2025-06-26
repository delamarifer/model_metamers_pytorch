#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=test_build_network_mms
#SBATCH --output=output/test_build_network_mms_style_%j.out
#SBATCH --error=output/test_build_network_mms_style_%j.err
#SBATCH --mem=16000
#SBATCH --time=4:00:00
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
cd /om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard

echo "=== Starting build_network.py Test (MMS Style) ==="
echo "Current directory: $(pwd)"
echo "This test uses the same pattern as make_mms.py to verify model performance"
echo "=================================================="

# Run the test script
echo "=== Running build_network.py Test (MMS Style) ==="
echo "This will test model loading, forward passes, and performance using make_mms.py patterns..."
echo "========================================================================================"

python test_build_network_mms_style.py

if [ $? -eq 0 ]; then
    echo "=== build_network.py Test (MMS Style) completed successfully ==="
    echo "✓ Your build_network.py is working correctly for metamer generation!"
    echo "✓ You can proceed with make_mms.py with confidence."
else
    echo "=== ERROR: build_network.py Test (MMS Style) failed! ==="
    echo "Check the error logs above for details."
    echo "✗ Please fix issues in build_network.py before proceeding with metamer generation."
    exit 1
fi

echo "=== Test Complete ==="
echo "Check the output files for detailed results:"
echo "  - Output: ./output/test_build_network_mms_style_${SLURM_JOB_ID}.out"
echo "  - Errors: ./output/test_build_network_mms_style_${SLURM_JOB_ID}.err"
echo "=============================================" 