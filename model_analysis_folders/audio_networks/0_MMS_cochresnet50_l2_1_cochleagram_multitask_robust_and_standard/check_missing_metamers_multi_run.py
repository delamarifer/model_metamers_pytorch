#!/usr/bin/env python3
"""
Script to check which parameter combinations are missing pickle files from metamer generation across multiple runs.
This helps identify which jobs failed and need to be re-run.
"""

import os
import glob
from pathlib import Path
import argparse

def check_missing_metamers_multi_run(base_dir, run_numbers):
    """
    Check which parameter combinations are missing pickle files across multiple runs.
    
    Args:
        base_dir: Base directory containing metamers_by_run
        run_numbers: List of run numbers to check (e.g., [41877180, 41883848])
    
    Returns:
        List of missing parameter combinations
    """
    
    # Define expected parameters based on the batch script
    sound_ids = list(range(37))  # 0-36
    model_types = ["robust", "standard"]
    random_seeds = [9, 400, 85]
    subclip_indices = [0, 1, 2]
    
    missing_combinations = []
    found_combinations = set()  # Track what we've found across all runs
    
    print("Checking multiple run directories: {}".format(run_numbers))
    
    # First, scan all runs to see what we have
    for run_number in run_numbers:
        run_dir = os.path.join(base_dir, "metamers_by_run", "metamers_{}".format(run_number))
        
        if not os.path.exists(run_dir):
            print("Warning: Run directory {} does not exist!".format(run_dir))
            continue
        
        print("Scanning run directory: {}".format(run_dir))
        
        # Check each combination in this run
        for model_type in model_types:
            for random_seed in random_seeds:
                # Construct the directory name pattern
                dir_pattern = "natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS{}_I3000_N8_LR1.000_DECAY0.500_{}".format(random_seed, model_type.upper())
                model_dir = os.path.join(run_dir, dir_pattern)
                
                if not os.path.exists(model_dir):
                    continue
                
                # Check each sound ID and subclip combination
                for sound_id in sound_ids:
                    for subclip_idx in subclip_indices:
                        # Construct the expected directory name
                        sound_dir_name = "{}_SOUND_about_{}_subclip{}".format(sound_id, model_type.upper(), subclip_idx)
                        sound_dir = os.path.join(model_dir, sound_dir_name)
                        
                        # Check if the pickle file exists
                        pickle_file = os.path.join(sound_dir, "all_metamers_pickle.pckl")
                        
                        if os.path.exists(pickle_file):
                            # Mark this combination as found
                            found_combinations.add((sound_id, model_type, random_seed, subclip_idx))
    
    # Now check what's missing
    print("\nChecking for missing combinations...")
    for model_type in model_types:
        for random_seed in random_seeds:
            for sound_id in sound_ids:
                for subclip_idx in subclip_indices:
                    if (sound_id, model_type, random_seed, subclip_idx) not in found_combinations:
                        missing_combinations.append({
                            'sound_id': sound_id,
                            'model_type': model_type,
                            'random_seed': random_seed,
                            'subclip_idx': subclip_idx
                        })
                        print("Missing: Sound {}, {}, seed {}, subclip {}".format(
                            sound_id, model_type, random_seed, subclip_idx))
    
    return missing_combinations

def generate_prioritized_rerun_script(missing_combinations, output_file="rerun_prioritized_metamers.sh"):
    """
    Generate a prioritized SLURM script that focuses on:
    1. Standard model first (more missing combinations)
    2. Complete coverage - getting all subclips for all sound IDs with at least one random seed
    
    Args:
        missing_combinations: List of missing parameter combinations
        output_file: Output script filename
    """
    
    # Separate by model type
    standard_missing = [c for c in missing_combinations if c['model_type'] == 'standard']
    robust_missing = [c for c in missing_combinations if c['model_type'] == 'robust']
    
    print("Standard model missing: {} combinations".format(len(standard_missing)))
    print("Robust model missing: {} combinations".format(len(robust_missing)))
    
    # For complete coverage, prioritize getting all subclips for each sound ID with seed 9 first
    # Then fill in the remaining gaps
    
    # Group standard combinations by sound ID and random seed
    standard_by_sound_seed = {}
    for combo in standard_missing:
        key = (combo['sound_id'], combo['random_seed'])
        if key not in standard_by_sound_seed:
            standard_by_sound_seed[key] = []
        standard_by_sound_seed[key].append(combo)
    
    # Prioritize: complete coverage with seed 9 first, then other seeds
    prioritized_standard = []
    
    # First, get all sound IDs that need complete coverage with seed 9
    seed_9_complete = []
    for sound_id in range(37):
        key = (sound_id, 9)
        if key in standard_by_sound_seed:
            # Check if we have all 3 subclips
            subclips = [c['subclip_idx'] for c in standard_by_sound_seed[key]]
            if len(subclips) == 3:  # All subclips missing
                seed_9_complete.extend(standard_by_sound_seed[key])
            elif len(subclips) < 3:  # Some subclips missing
                seed_9_complete.extend(standard_by_sound_seed[key])
    
    # Add remaining standard combinations
    remaining_standard = []
    for combo in standard_missing:
        if combo not in seed_9_complete:
            remaining_standard.append(combo)
    
    # Group robust combinations
    robust_by_sound_seed = {}
    for combo in robust_missing:
        key = (combo['sound_id'], combo['random_seed'])
        if key not in robust_by_sound_seed:
            robust_by_sound_seed[key] = []
        robust_by_sound_seed[key].append(combo)
    
    # Prioritize robust: complete coverage with seed 9 first
    prioritized_robust = []
    seed_9_robust_complete = []
    for sound_id in range(37):
        key = (sound_id, 9)
        if key in robust_by_sound_seed:
            subclips = [c['subclip_idx'] for c in robust_by_sound_seed[key]]
            if len(subclips) > 0:  # Any subclips missing
                seed_9_robust_complete.extend(robust_by_sound_seed[key])
    
    # Add remaining robust combinations
    remaining_robust = []
    for combo in robust_missing:
        if combo not in seed_9_robust_complete:
            remaining_robust.append(combo)
    
    # Create prioritized list
    prioritized_combinations = seed_9_complete + remaining_standard + seed_9_robust_complete + remaining_robust
    
    print("\nPrioritized order:")
    print("1. Standard model - complete coverage with seed 9: {} combinations".format(len(seed_9_complete)))
    print("2. Standard model - remaining: {} combinations".format(len(remaining_standard)))
    print("3. Robust model - complete coverage with seed 9: {} combinations".format(len(seed_9_robust_complete)))
    print("4. Robust model - remaining: {} combinations".format(len(remaining_robust)))
    
    # Group by model type and random seed for efficient array jobs
    grouped_combinations = {}
    
    for combo in prioritized_combinations:
        key = (combo['model_type'], combo['random_seed'])
        if key not in grouped_combinations:
            grouped_combinations[key] = []
        grouped_combinations[key].append(combo)
    
    script_content = """#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_prioritized
#SBATCH --output=output/prioritized_metamers_%A_%a.out
#SBATCH --error=output/prioritized_metamers_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-{}
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
""".format(len(grouped_combinations)-1)
    
    # Add each combination group
    for i, ((model_type, random_seed), combos) in enumerate(grouped_combinations.items()):
        sound_ids = sorted(list(set(combo['sound_id'] for combo in combos)))
        subclip_indices = sorted(list(set(combo['subclip_idx'] for combo in combos)))
        
        script_content += '    "{} {} {} {}"\n'.format(
            model_type, 
            random_seed, 
            " ".join(map(str, sound_ids)), 
            " ".join(map(str, subclip_indices))
        )
    
    script_content += """)

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
UNIQUE_RUN_NUMBER=$((SLURM_ARRAY_JOB_ID + 3000000))  # Offset to avoid conflicts
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using new run number: $UNIQUE_RUN_NUMBER"
echo "METAMER_RUN_NUMBER environment variable: $METAMER_RUN_NUMBER"

# Loop through sound IDs and subclip indices
for SOUND_ID in "${SOUND_IDS[@]}"; do
    for SUBCLIP_IDX in "${SUBCLIP_INDICES[@]}"; do
        echo "=== Processing sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        
        # Create model-specific output directory
        OUTPUT_DIR="plots/prioritized_metamers_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== Directory Setup ==="
        echo "Output Directory: $OUTPUT_DIR"
        echo "Current Directory: $(pwd)"
        echo "======================"

        # Build the command
        CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

        echo "=== Running Metamer Generation for $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
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
            echo "Error: Metamer generation failed for $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED with exit code $?"
            continue
        fi

        echo "=== Successfully completed $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
    done
done

echo "=== All Processing Complete ==="
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print("Generated prioritized rerun script: {}".format(output_file))
    print("Total array tasks: {}".format(len(grouped_combinations)))

def generate_rerun_script(missing_combinations, output_file="rerun_missing_metamers.sh"):
    """
    Generate a SLURM script to rerun only the missing combinations.
    
    Args:
        missing_combinations: List of missing parameter combinations
        output_file: Output script filename
    """
    
    # Group by model type and random seed to create efficient array jobs
    grouped_combinations = {}
    
    for combo in missing_combinations:
        key = (combo['model_type'], combo['random_seed'])
        if key not in grouped_combinations:
            grouped_combinations[key] = []
        grouped_combinations[key].append(combo)
    
    script_content = """#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_rerun_missing
#SBATCH --output=output/rerun_metamers_%A_%a.out
#SBATCH --error=output/rerun_metamers_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-{}
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
""".format(len(grouped_combinations)-1)
    
    # Add each combination group
    for i, ((model_type, random_seed), combos) in enumerate(grouped_combinations.items()):
        sound_ids = sorted(list(set(combo['sound_id'] for combo in combos)))
        subclip_indices = sorted(list(set(combo['subclip_idx'] for combo in combos)))
        
        script_content += '    "{} {} {} {}"\n'.format(
            model_type, 
            random_seed, 
            " ".join(map(str, sound_ids)), 
            " ".join(map(str, subclip_indices))
        )
    
    script_content += """)

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
        OUTPUT_DIR="plots/rerun_metamers_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== Directory Setup ==="
        echo "Output Directory: $OUTPUT_DIR"
        echo "Current Directory: $(pwd)"
        echo "======================"

        # Build the command
        CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

        echo "=== Running Metamer Generation for $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
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
            echo "Error: Metamer generation failed for $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED with exit code $?"
            continue
        fi

        echo "=== Successfully completed $MODEL_TYPE, sound $SOUND_ID, subclip $SUBCLIP_IDX, seed $RANDOM_SEED ==="
    done
done

echo "=== All Processing Complete ==="
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print("Generated rerun script: {}".format(output_file))
    print("Total array tasks: {}".format(len(grouped_combinations)))

def main():
    parser = argparse.ArgumentParser(description='Check for missing metamers pickle files across multiple runs and generate rerun script')
    parser.add_argument('--base_dir', type=str, 
                       default='/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard',
                       help='Base directory containing metamers_by_run')
    parser.add_argument('--run_numbers', type=str, nargs='+', required=True,
                       help='Run numbers to check (e.g., 41877180 41883848)')
    parser.add_argument('--output_script', type=str, default='rerun_missing_metamers.sh',
                       help='Output script filename')
    parser.add_argument('--prioritized', action='store_true',
                       help='Generate prioritized script focusing on standard model and complete coverage')
    
    args = parser.parse_args()
    
    print("Checking for missing metamers across runs: {}".format(args.run_numbers))
    print("Base directory: {}".format(args.base_dir))
    
    # Check for missing combinations across all runs
    missing_combinations = check_missing_metamers_multi_run(args.base_dir, args.run_numbers)
    
    print("\n=== Summary ===")
    print("Total missing combinations: {}".format(len(missing_combinations)))
    
    if missing_combinations:
        print("\nMissing combinations:")
        for combo in missing_combinations:
            print("  Sound {}, {}, seed {}, subclip {}".format(
                combo['sound_id'], combo['model_type'], combo['random_seed'], combo['subclip_idx']))
        
        # Generate appropriate rerun script
        if args.prioritized:
            generate_prioritized_rerun_script(missing_combinations, args.output_script)
        else:
            generate_rerun_script(missing_combinations, args.output_script)
        
        print("\nTo rerun missing jobs, use:")
        print("sbatch {}".format(args.output_script))
    else:
        print("No missing combinations found! All jobs completed successfully.")

if __name__ == "__main__":
    main() 