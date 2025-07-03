#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_nhm2018_timeavg_standard
#SBATCH --output=output/metamers_%A_%a.out
#SBATCH --error=output/metamers_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-35 # Array indices for standard model (sound IDs 0-36)
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

# Default to no subclip if not specified
SUBCLIP_IDX=${1:-}
# Default duration to 3 seconds if not specified
DURATION=${2:-3}

echo "=== Configuration ==="
echo "Subclip Index: $SUBCLIP_IDX"
echo "Duration: $DURATION"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "===================="

# Validate duration
if [[ ! "$DURATION" =~ ^(2|3|4|7|10)$ ]]; then
    echo "Error: Duration must be one of: 2, 3, 4, 7, or 10 seconds"
    exit 1
fi

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="plots/metamers_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=== Directory Setup ==="
echo "Output Directory: $OUTPUT_DIR"
echo "Current Directory: $(pwd)"
echo "======================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Loop over subclip indices
for SUBCLIP_IDX in 0 1 2; do
    echo "=== Processing subclip $SUBCLIP_IDX ==="
    CMD="python make_mms.py $SLURM_ARRAY_TASK_ID -I 3000 -N 8 -F natural_sounds_norman_haignere --duration 3 -L coarse_define_spectemp_inversion_loss_layer --subclip_idx $SUBCLIP_IDX"
    echo "Command: $CMD"
    $CMD

    if [ $? -ne 0 ]; then
        echo "Error: Metamer generation failed for subclip $SUBCLIP_IDX with exit code $?"
        continue
    fi

done

echo "=== All subclips processed ==="

echo "=== Finding Metamer Directory ==="
# Find the most recent metamer directory
METAMER_DIR=$(find . -type d -path "*/metamers/*" -printf "%T@ %p\n" | sort -n | tail -1 | cut -f2- -d" ")

if [ -z "$METAMER_DIR" ]; then
    echo "Error: No metamer directory found!"
    echo "Current directory contents:"
    ls -R
    exit 1
fi

echo "Found metamer directory: $METAMER_DIR"
echo "Directory contents:"
ls -l "$METAMER_DIR"
echo "================================"

echo "=== Analysis Complete ==="
echo "Results saved in: $OUTPUT_DIR"
echo "========================"


