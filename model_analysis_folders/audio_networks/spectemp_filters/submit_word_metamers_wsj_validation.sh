#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_nhm2018_timeavg_debug
#SBATCH --output=output/robust_timeavg_debug%A_%a.out
#SBATCH --error=output/robust_timeavg_debug%A_%a.err
#SBATCH --mem=4000
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0  # Adjust this based on number of files you want to process
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

# Build the command with the correct arguments
CMD="python make_metamers_wsj400_behavior_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -I 10 -N 1 -L coarse_define_spectemp_inversion_loss_layer -F natural_sounds_norman_haignere --duration $DURATION --debug_loss"
if [ ! -z "$SUBCLIP_IDX" ]; then
    CMD="$CMD --subclip_idx $SUBCLIP_IDX"
fi

echo "=== Running Metamer Generation ==="
echo "Command: $CMD"
echo "================================="

# Run the metamer generation
$CMD

# Check if metamer generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Metamer generation failed with exit code $?"
    exit 1
fi

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


