#!/bin/bash -l

# Base directory containing the metamer folders
METAMERS_DIR="metamers"
PLOTS_DIR="plots"
ERROR_LOG="pickle_errors.log"

# Check if GPU ID is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    echo "Example: $0 0  # to use GPU 0"
    exit 1
fi

GPU_ID=$1
echo "Using GPU $GPU_ID"

# Enable error handling
set -e
set -o pipefail

# Load CUDA module if available
module load cuda70/toolkit/7.0.28

# Activate conda environment
source ~/.bashrc
conda activate model_metamers_pytorch

# Get the repository root directory
REPO_ROOT="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch"

# Add the repository root and analysis_scripts to PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/analysis_scripts:$PYTHONPATH

# Create plots directory if it doesn't exist
mkdir -p "$PLOTS_DIR"

# Initialize error log
echo "=== Pickle Processing Errors ===" > "$ERROR_LOG"
echo "Started at: $(date)" >> "$ERROR_LOG"
echo "=============================" >> "$ERROR_LOG"

# Print GPU information for debugging
echo "=== GPU Information ==="
nvidia-smi
echo "======================"

# Count total directories for progress tracking
total_dirs=$(find "$METAMERS_DIR" -type d -name "*_RS*_I*_N*_*_*" | wc -l)
current_dir=0

# Find all metamer directories
find "$METAMERS_DIR" -type d -name "*_RS*_I*_N*_*_*" | while read -r metamer_dir; do
    current_dir=$((current_dir + 1))
    
    # Extract parameters from directory name
    # Example: natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS42_I3000_N8_ROBUST_20250616_100511
    if [[ $metamer_dir =~ RS([0-9]+)_I[0-9]+_N([0-9]+)_(ROBUST|STANDARD)_[0-9]+_[0-9]+ ]]; then
        seed="${BASH_REMATCH[1]}"
        model_type="${BASH_REMATCH[3],,}"  # Convert to lowercase
        
        echo "Processing directory $current_dir of $total_dirs:"
        echo "  Directory: $metamer_dir"
        echo "  Seed: $seed"
        echo "  Model Type: $model_type"
        
        # Find all subdirectories containing pickles
        find "$metamer_dir" -type d -name "*_SOUND_*_subclip*" | while read -r subdir; do
            if [ -f "$subdir/all_metamers_pickle.pckl" ]; then
                # Extract sound ID from subdirectory name
                # Example: 0_SOUND_about_ROBUST_subclip0
                if [[ $subdir =~ ([0-9]+)_SOUND_ ]]; then
                    sound_id="${BASH_REMATCH[1]}"
                    echo "  Found pickle in: $subdir (Sound ID: $sound_id)"
                    
                    # Run the plotting script with GPU assignment and error handling
                    if ! CUDA_VISIBLE_DEVICES=$GPU_ID python make_single_layer_plots_cochmanual_mse.py \
                        --base_path "$subdir" \
                        --output_folder "$PLOTS_DIR" \
                        --sound_id "$sound_id" \
                        --rand_seed_1 "$seed" \
                        --model_type "$model_type" \
                        --loss_type "inversion_loss_layer" 2>> "$ERROR_LOG"; then
                        
                        echo "  [ERROR] Failed to process pickle in $subdir"
                        echo "  Check $ERROR_LOG for details"
                    fi
                fi
            fi
        done
        
        echo "----------------------------------------"
    else
        echo "Warning: Could not parse parameters from directory: $metamer_dir"
    fi
done

echo "All processing complete!"
echo "Check $ERROR_LOG for any errors that occurred during processing." 