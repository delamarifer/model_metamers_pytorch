#!/usr/bin/env python3
"""
Script to check completion status of metamer generation jobs.
Scans the metamers directory and reports which pickle files were created vs missing.
"""

import os
import glob
import argparse
from collections import defaultdict

def check_completion_status(run_number, sound_ids=None, model_types=None, subclips=None, random_seeds=None):
    """
    Check which metamer generation jobs completed successfully.
    
    Args:
        run_number (int): The run number to check
        sound_ids (list): List of sound IDs to check (default: all found)
        model_types (list): List of model types to check (default: ['standard', 'robust'])
        subclips (list): List of subclip indices to check (default: [0, 1, 2])
        random_seeds (list): List of random seeds to check (default: [42, 123, 456])
    """
    
    # Default values
    if model_types is None:
        model_types = ['standard', 'robust']
    if subclips is None:
        subclips = [0, 1, 2]
    if random_seeds is None:
        random_seeds = [42, 123, 456]
    
    # Base directory for metamers
    metamers_base = f"metamers_{run_number}"
    
    if not os.path.exists(metamers_base):
        print(f"Error: Metamers directory '{metamers_base}' not found!")
        return
    
    print(f"=== Checking completion status for run {run_number} ===")
    print(f"Base directory: {metamers_base}")
    print()
    
    # Find all sound directories
    sound_dirs = glob.glob(os.path.join(metamers_base, "*_SOUND_*"))
    
    if not sound_dirs:
        print("No sound directories found!")
        return
    
    # Extract sound IDs from directory names
    found_sound_ids = []
    for sound_dir in sound_dirs:
        try:
            # Extract sound ID from directory name like "5_SOUND_word_robust_subclip0"
            parts = os.path.basename(sound_dir).split('_')
            sound_id = int(parts[0])
            found_sound_ids.append(sound_id)
        except (ValueError, IndexError):
            continue
    
    if sound_ids is None:
        sound_ids = sorted(set(found_sound_ids))
    
    print(f"Found sound IDs: {sorted(found_sound_ids)}")
    print(f"Checking sound IDs: {sound_ids}")
    print()
    
    # Track completion status
    completion_status = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for sound_id in sound_ids:
        print(f"Sound ID {sound_id}:")
        
        for model_type in model_types:
            print(f"  {model_type.upper()} model:")
            
            for subclip_idx in subclips:
                print(f"    Subclip {subclip_idx}:")
                
                for random_seed in random_seeds:
                    # Construct expected pickle file path
                    expected_pickle_path = os.path.join(
                        metamers_base,
                        f"{sound_id}_SOUND_word_{model_type.upper()}_subclip{subclip_idx}",
                        "all_metamers_pickle.pckl"
                    )
                    
                    # Check if file exists
                    exists = os.path.exists(expected_pickle_path)
                    completion_status[sound_id][model_type][subclip_idx][random_seed] = exists
                    
                    status = "✓ COMPLETED" if exists else "✗ MISSING"
                    print(f"      Seed {random_seed}: {status}")
                    
                    if exists:
                        # Get file size for additional info
                        file_size = os.path.getsize(expected_pickle_path)
                        print(f"        File size: {file_size:,} bytes")
        
        print()
    
    # Generate summary
    print("=== SUMMARY ===")
    total_expected = len(sound_ids) * len(model_types) * len(subclips) * len(random_seeds)
    total_completed = 0
    total_missing = 0
    
    for sound_id in sound_ids:
        for model_type in model_types:
            for subclip_idx in subclips:
                completed_count = sum(1 for seed in random_seeds 
                                    if completion_status[sound_id][model_type][subclip_idx][seed])
                missing_count = len(random_seeds) - completed_count
                total_completed += completed_count
                total_missing += missing_count
                
                if missing_count > 0:
                    missing_seeds = [seed for seed in random_seeds 
                                   if not completion_status[sound_id][model_type][subclip_idx][seed]]
                    print(f"Sound {sound_id}, {model_type.upper()}, Subclip {subclip_idx}: "
                          f"{completed_count}/{len(random_seeds)} completed "
                          f"(missing seeds: {missing_seeds})")
    
    print()
    print(f"Overall completion: {total_completed}/{total_expected} ({total_completed/total_expected*100:.1f}%)")
    print(f"Completed: {total_completed}")
    print(f"Missing: {total_missing}")
    
    # Generate resubmission commands for missing jobs
    if total_missing > 0:
        print()
        print("=== RESUBMISSION COMMANDS ===")
        print("# Use these commands to resubmit missing jobs:")
        
        for sound_id in sound_ids:
            for model_type in model_types:
                for subclip_idx in subclips:
                    missing_seeds = [seed for seed in random_seeds 
                                   if not completion_status[sound_id][model_type][subclip_idx][seed]]
                    
                    if missing_seeds:
                        for seed in missing_seeds:
                            cmd = (f"python make_mms.py {sound_id} -I 3000 -N 8 -M {model_type} "
                                   f"-F natural_sounds_norman_haignere --duration 3 "
                                   f"-L time_averaged_inversion_loss_layer --subclip_idx {subclip_idx} "
                                   f"-Z 0.1 --lr_decay 0.7 -R {seed}")
                            print(f"# Sound {sound_id}, {model_type.upper()}, Subclip {subclip_idx}, Seed {seed}:")
                            print(f"{cmd}")
                            print()

def main():
    parser = argparse.ArgumentParser(description='Check completion status of metamer generation jobs')
    parser.add_argument('run_number', type=int, help='Run number to check')
    parser.add_argument('--sound-ids', nargs='+', type=int, help='Specific sound IDs to check')
    parser.add_argument('--model-types', nargs='+', choices=['standard', 'robust'], 
                       help='Model types to check')
    parser.add_argument('--subclips', nargs='+', type=int, help='Subclip indices to check')
    parser.add_argument('--random-seeds', nargs='+', type=int, help='Random seeds to check')
    
    args = parser.parse_args()
    
    check_completion_status(
        run_number=args.run_number,
        sound_ids=args.sound_ids,
        model_types=args.model_types,
        subclips=args.subclips,
        random_seeds=args.random_seeds
    )

if __name__ == '__main__':
    main() 