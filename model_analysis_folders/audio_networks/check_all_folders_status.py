#!/usr/bin/env python3
"""
Script to check completion status across all audio network folders.
This scans all folders under audio_networks and checks their metamers_by_run subdirectories
to see which combinations are completed.
"""

import os
import glob
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import json

def get_audio_network_folders(base_dir):
    """Get all audio network folders that contain metamers_by_run subdirectories."""
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            metamers_path = os.path.join(item_path, "metamers_by_run")
            if os.path.exists(metamers_path):
                folders.append(item)
    return sorted(folders)

def check_folder_status(folder_path, folder_name):
    """
    Check the completion status for a single folder.
    
    Args:
        folder_path: Full path to the folder
        folder_name: Name of the folder
    
    Returns:
        Dictionary with completion statistics
    """
    metamers_path = os.path.join(folder_path, "metamers_by_run")
    
    if not os.path.exists(metamers_path):
        return {
            'folder': folder_name,
            'status': 'no_metamers_dir',
            'runs': [],
            'total_expected': 0,
            'total_found': 0,
            'completion_rate': 0.0
        }
    
    # Find all run directories
    run_dirs = []
    for item in os.listdir(metamers_path):
        if item.startswith("metamers_"):
            run_number = item.replace("metamers_", "")
            run_dirs.append(run_number)
    
    if not run_dirs:
        return {
            'folder': folder_name,
            'status': 'no_runs',
            'runs': [],
            'total_expected': 0,
            'total_found': 0,
            'completion_rate': 0.0
        }
    
    # Define expected parameters (based on the existing scripts)
    sound_ids = list(range(37))  # 0-36
    model_types = ["robust", "standard"]
    random_seeds = [9, 400, 85]
    subclip_indices = [0, 1, 2]
    
    total_expected = len(sound_ids) * len(model_types) * len(random_seeds) * len(subclip_indices)
    total_found = 0
    run_details = []
    
    for run_number in sorted(run_dirs):
        run_dir = os.path.join(metamers_path, f"metamers_{run_number}")
        run_found = 0
        run_expected = total_expected
        
        # Check each combination in this run
        for model_type in model_types:
            for random_seed in random_seeds:
                # Construct the directory name pattern
                dir_pattern = f"natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS{random_seed}_I3000_N8_LR1.000_DECAY0.500_{model_type.upper()}"
                model_dir = os.path.join(run_dir, dir_pattern)
                
                if not os.path.exists(model_dir):
                    continue
                
                # Check each sound ID and subclip combination
                for sound_id in sound_ids:
                    for subclip_idx in subclip_indices:
                        # Construct the expected directory name
                        sound_dir_name = f"{sound_id}_SOUND_about_{model_type.upper()}_subclip{subclip_idx}"
                        sound_dir = os.path.join(model_dir, sound_dir_name)
                        
                        # Check if the pickle file exists
                        pickle_file = os.path.join(sound_dir, "all_metamers_pickle.pckl")
                        
                        if os.path.exists(pickle_file):
                            run_found += 1
                            total_found += 1
        
        run_details.append({
            'run_number': run_number,
            'found': run_found,
            'expected': run_expected,
            'completion_rate': run_found / run_expected if run_expected > 0 else 0.0
        })
    
    completion_rate = total_found / total_expected if total_expected > 0 else 0.0
    
    return {
        'folder': folder_name,
        'status': 'has_runs',
        'runs': run_details,
        'total_expected': total_expected,
        'total_found': total_found,
        'completion_rate': completion_rate
    }

def print_summary(all_results, verbose=False):
    """Print a summary of all folder results."""
    print("=" * 80)
    print("COMPLETION STATUS SUMMARY ACROSS ALL AUDIO NETWORK FOLDERS")
    print("=" * 80)
    
    # Overall statistics
    total_folders = len(all_results)
    folders_with_runs = sum(1 for r in all_results if r['status'] == 'has_runs')
    folders_with_no_runs = sum(1 for r in all_results if r['status'] == 'no_runs')
    folders_with_no_metamers = sum(1 for r in all_results if r['status'] == 'no_metamers_dir')
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total folders checked: {total_folders}")
    print(f"  Folders with runs: {folders_with_runs}")
    print(f"  Folders with no runs: {folders_with_no_runs}")
    print(f"  Folders with no metamers_by_run: {folders_with_no_metamers}")
    
    # Completion rates for folders with runs
    if folders_with_runs > 0:
        completion_rates = [r['completion_rate'] for r in all_results if r['status'] == 'has_runs']
        avg_completion = sum(completion_rates) / len(completion_rates)
        print(f"  Average completion rate: {avg_completion:.1%}")
        
        # Find best and worst performing folders
        completed_folders = [r for r in all_results if r['status'] == 'has_runs']
        completed_folders.sort(key=lambda x: x['completion_rate'], reverse=True)
        
        print(f"\nTOP 5 MOST COMPLETE FOLDERS:")
        for i, result in enumerate(completed_folders[:5]):
            print(f"  {i+1}. {result['folder']}: {result['completion_rate']:.1%} ({result['total_found']}/{result['total_expected']})")
        
        print(f"\nBOTTOM 5 LEAST COMPLETE FOLDERS:")
        for i, result in enumerate(completed_folders[-5:]):
            print(f"  {i+1}. {result['folder']}: {result['completion_rate']:.1%} ({result['total_found']}/{result['total_expected']})")
    
    # Detailed results if verbose
    if verbose:
        print(f"\nDETAILED RESULTS:")
        for result in all_results:
            print(f"\n{result['folder']}:")
            if result['status'] == 'no_metamers_dir':
                print(f"  Status: No metamers_by_run directory")
            elif result['status'] == 'no_runs':
                print(f"  Status: No run directories found")
            else:
                print(f"  Status: {result['total_found']}/{result['total_expected']} combinations completed ({result['completion_rate']:.1%})")
                for run in result['runs']:
                    print(f"    Run {run['run_number']}: {run['found']}/{run['expected']} ({run['completion_rate']:.1%})")

def save_results_to_json(all_results, output_file):
    """Save results to a JSON file for further analysis."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Check completion status across all audio network folders')
    parser.add_argument('--base-dir', default='.', help='Base directory containing audio network folders')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed results')
    parser.add_argument('--output', '-o', help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Get the audio networks directory
    audio_networks_dir = os.path.join(args.base_dir, "model_analysis_folders", "audio_networks")
    
    if not os.path.exists(audio_networks_dir):
        print(f"Error: Audio networks directory not found: {audio_networks_dir}")
        return
    
    print(f"Scanning audio network folders in: {audio_networks_dir}")
    
    # Get all folders
    folders = get_audio_network_folders(audio_networks_dir)
    print(f"Found {len(folders)} folders with metamers_by_run directories:")
    for folder in folders:
        print(f"  - {folder}")
    
    # Check each folder
    all_results = []
    for folder in folders:
        folder_path = os.path.join(audio_networks_dir, folder)
        result = check_folder_status(folder_path, folder)
        all_results.append(result)
    
    # Print summary
    print_summary(all_results, verbose=args.verbose)
    
    # Save to JSON if requested
    if args.output:
        save_results_to_json(all_results, args.output)

if __name__ == "__main__":
    main() 