#!/usr/bin/env python3
"""
Script to merge all metamer pickle files from different runs into a single folder
with consistent naming that preserves all important information.

Usage:
    python merge_metamer_pickles.py [--output_dir merged_metamers] [--dry_run]
"""

import os
import shutil
import re
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import pickle

def parse_metamer_path(filepath):
    """
    Parse a metamer pickle file path to extract all important information.
    
    Expected path structure:
    metamers_by_run/metamers_{run_number}/{experiment_name}/{sound_id}_SOUND_{word}_{model_type}_subclip{subclip_idx}/all_metamers_pickle.pckl
    
    Returns a dictionary with all parsed information.
    """
    path = Path(filepath)
    
    # Extract run number from parent directory
    run_match = re.search(r'metamers_(\d+)', str(path))
    run_number = run_match.group(1) if run_match else "unknown_run"
    
    # Get the experiment directory name (contains all the parameters)
    experiment_dir = path.parent.parent.name
    
    # Get the sound directory name
    sound_dir = path.parent.name
    # Example: 14_SOUND_about_ROBUST_subclip0
    
    # Parse sound directory
    sound_match = re.match(r'(\d+)_SOUND_(\w+)_(\w+)_subclip(\d+)', sound_dir)
    if sound_match:
        sound_id = sound_match.group(1)
        subclip_idx = sound_match.group(4)
    else:
        # Fallback parsing
        parts = sound_dir.split('_')
        sound_id = parts[0] if parts else "unknown"
        subclip_idx = parts[-1] if parts else "0"
    
    return {
        'run_number': run_number,
        'experiment_dir': experiment_dir,
        'sound_id': sound_id,
        'subclip_idx': subclip_idx,
        'original_path': str(filepath)
    }

def generate_consistent_filename(parsed_info):
    """
    Generate a consistent filename from parsed information.
    
    Format: run_{run_number}_{experiment_dir}_SOUNDID_{sound_id}_SUBCLIP{subclip_idx}.pckl
    """
    filename = (
        f"run_{parsed_info['run_number']}_"
        f"{parsed_info['experiment_dir']}_"
        f"SOUNDID_{parsed_info['sound_id']}_"
        f"SUBCLIP{parsed_info['subclip_idx']}.pckl"
    )
    return filename

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def merge_metamer_pickles(base_dir, output_dir, dry_run=False):
    """
    Find all metamer pickle files and merge them into a single directory with consistent naming.
    
    Args:
        base_dir: Base directory containing metamers_by_run
        output_dir: Output directory for merged files
        dry_run: If True, only print what would be done without actually copying files
    """
    metamers_dir = os.path.join(base_dir, 'metamers_by_run')
    
    if not os.path.exists(metamers_dir):
        print(f"Error: metamers_by_run directory not found at {metamers_dir}")
        return
    
    # Find all pickle files
    pickle_files = []
    for root, dirs, files in os.walk(metamers_dir):
        for file in files:
            if file == 'all_metamers_pickle.pckl':
                pickle_files.append(os.path.join(root, file))
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Parse each file and generate new names
    parsed_files = []
    for filepath in pickle_files:
        try:
            parsed_info = parse_metamer_path(filepath)
            new_filename = generate_consistent_filename(parsed_info)
            parsed_files.append({
                'original_path': filepath,
                'new_filename': new_filename,
                'parsed_info': parsed_info
            })
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            continue
    
    # Create output directory
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    # Track filename usage to handle duplicates
    filename_counts = {}
    successful_copies = []
    
    for file_info in parsed_files:
        original_path = file_info['original_path']
        base_filename = file_info['new_filename']
        
        # Handle duplicate filenames by adding suffix
        if base_filename in filename_counts:
            filename_counts[base_filename] += 1
            # Remove .pckl extension, add suffix, then add .pckl back
            name_without_ext = base_filename[:-5]  # Remove .pckl
            new_filename = f"{name_without_ext}_copy{filename_counts[base_filename]}.pckl"
        else:
            filename_counts[base_filename] = 0
            new_filename = base_filename
        
        new_path = os.path.join(output_dir, new_filename)
        
        # Copy the file
        if dry_run:
            print(f"Would copy: {original_path} -> {new_path}")
        else:
            try:
                shutil.copy2(original_path, new_path)
                successful_copies.append({
                    'original': original_path,
                    'new': new_path,
                    'filename': new_filename
                })
                print(f"Copied: {new_filename}")
            except Exception as e:
                print(f"Error copying {original_path}: {e}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total files found: {len(pickle_files)}")
    print(f"Successfully parsed: {len(parsed_files)}")
    print(f"Successfully copied: {len(successful_copies)}")
    
    # Count duplicates
    duplicates = sum(1 for count in filename_counts.values() if count > 0)
    if duplicates > 0:
        print(f"Files with duplicate names (suffixes added): {duplicates}")
        print(f"\n=== Duplicate Files ===")
        for filename, count in filename_counts.items():
            if count > 0:
                print(f"  {filename} -> {count} copies with suffixes")
    
    if not dry_run:
        print(f"\nAll files merged to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Merge metamer pickle files from different runs')
    parser.add_argument('--base_dir', default='.', 
                       help='Base directory containing metamers_by_run (default: current directory)')
    parser.add_argument('--output_dir', default='merged_metamers',
                       help='Output directory for merged files (default: merged_metamers)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be done without actually copying files')
    
    args = parser.parse_args()
    
    # Use absolute paths
    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    merge_metamer_pickles(base_dir, output_dir, args.dry_run)

if __name__ == "__main__":
    main() 