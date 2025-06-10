import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
import logging
import argparse
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_10s_data(file_path):
    """
    Load the 10-second interval dataset.
    
    Args:
        file_path (str): Path to the CSV file containing 10s data
        
    Returns:
        pandas.DataFrame: Loaded data with datetime index
    """
    try:
        df = pd.read_csv(file_path)
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def resample_to_hourly(df):
    """
    Resample 10-second data to hourly intervals.
    
    Args:
        df (pandas.DataFrame): Input dataframe with 10s intervals
        
    Returns:
        pandas.DataFrame: Hourly resampled data
    """
    # Resample to hourly intervals using mean
    hourly_data = df.resample('H').mean()
    return hourly_data

def resample_to_daily(df):
    """
    Resample 10-second data to daily intervals.
    
    Args:
        df (pandas.DataFrame): Input dataframe with 10s intervals
        
    Returns:
        pandas.DataFrame: Daily resampled data
    """
    # Resample to daily intervals using mean
    daily_data = df.resample('D').mean()
    return daily_data

def calculate_rolling_statistics(df, window='1H'):
    """
    Calculate rolling statistics for the data.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        window (str): Rolling window size (e.g., '1H' for 1 hour)
        
    Returns:
        pandas.DataFrame: Data with rolling statistics
    """
    # Calculate rolling mean and standard deviation
    rolling_stats = pd.DataFrame()
    for column in df.select_dtypes(include=[np.number]).columns:
        rolling_stats[f'{column}_rolling_mean'] = df[column].rolling(window=window).mean()
        rolling_stats[f'{column}_rolling_std'] = df[column].rolling(window=window).std()
    
    return rolling_stats

def save_restructured_data(df, output_path, format='csv'):
    """
    Save the restructured data to a file.
    
    Args:
        df (pandas.DataFrame): Data to save
        output_path (str): Path to save the data
        format (str): Output format ('csv' or 'parquet')
    """
    if format.lower() == 'csv':
        df.to_csv(output_path)
    elif format.lower() == 'parquet':
        df.to_parquet(output_path)
    else:
        raise ValueError("Format must be either 'csv' or 'parquet'")

def get_reference_structure(base_dir, reference_duration='2SECS'):
    """
    Get the folder structure from a reference duration folder.
    
    Args:
        base_dir (str): Base directory containing all duration folders
        reference_duration (str): Duration folder to use as reference (e.g., '2SECS', '3SECS', '4SECS')
        
    Returns:
        list: List of subdirectory names from the reference structure
    """
    reference_path = os.path.join(base_dir, f"{reference_duration}_Norman-Haignere_McDermott_2018_Stimuli/NATURAL")
    
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference path does not exist: {reference_path}")
    
    # Get all subdirectories in the NATURAL folder
    subdirs = [d for d in os.listdir(reference_path) 
              if os.path.isdir(os.path.join(reference_path, d))]
    
    # Sort the directories to maintain consistent order
    subdirs.sort()
    
    logger.info(f"Found {len(subdirs)} subdirectories in reference structure")
    return subdirs

def create_directory_structure(base_path, reference_subdirs, dry_run=False):
    """
    Create the NATURAL directory and its subdirectories to match the structure
    of the reference duration dataset.
    
    Args:
        base_path (str): Base path for the new structure
        reference_subdirs (list): List of subdirectory names from reference structure
        dry_run (bool): If True, only show what would be done without making changes
    """
    # Create NATURAL directory
    natural_dir = os.path.join(base_path, 'NATURAL')
    if not dry_run:
        os.makedirs(natural_dir, exist_ok=True)
    logger.info(f"{'Would create' if dry_run else 'Created'} NATURAL directory at {natural_dir}")
    
    # Create all subdirectories
    for subdir in reference_subdirs:
        full_path = os.path.join(natural_dir, subdir)
        if not dry_run:
            os.makedirs(full_path, exist_ok=True)
        logger.info(f"{'Would create' if dry_run else 'Created'} directory: {full_path}")

def get_stimulus_number(filename):
    """
    Extract the stimulus number from the filename.
    Example: '0_stim102_crumpling_paper_orig.wav' -> '0'
    """
    try:
        # Split by underscore and get the first part
        return filename.split('_')[0]
    except Exception as e:
        logger.error(f"Error extracting stimulus number from {filename}: {e}")
        return None

def add_duration_suffix(filename, duration='10'):
    """
    Add duration suffix to the filename.
    Example: '0_stim102_crumpling_paper_orig.wav' -> '0_stim102_crumpling_paper_orig_00_10.wav'
    """
    # Remove .wav extension
    base_name = os.path.splitext(filename)[0]
    # Add duration suffix and .wav extension
    return f"{base_name}_00_{duration}.wav"

def find_wav_files(source_dir):
    """
    Find all WAV files in the source directory and its subdirectories.
    Returns a list of (file_path, filename) tuples.
    """
    wav_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append((full_path, file))
    return wav_files

def reorganize_files(source_dir, target_dir, reference_subdirs, dry_run=False):
    """
    Reorganize files from the source directory to match the structure
    of the reference duration dataset.
    """
    # Create the new directory structure
    create_directory_structure(target_dir, reference_subdirs, dry_run)
    
    # Get the source directory path
    source_path = Path(source_dir)
    
    # Track statistics
    total_files = 0
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    # Find all WAV files
    logger.info(f"Searching for WAV files in: {source_path}")
    wav_files = find_wav_files(source_path)
    total_files = len(wav_files)
    logger.info(f"Found {total_files} WAV files")
    
    # Create a mapping of stimulus numbers to target directories
    target_dirs = {subdir.split('_')[0]: subdir for subdir in reference_subdirs}
    
    for source_file, filename in wav_files:
        # Skip any hidden files
        if filename.startswith('.'):
            logger.debug(f"Skipping hidden file: {filename}")
            skipped_files += 1
            continue
            
        try:
            # Extract the stimulus number from the filename
            stim_num = get_stimulus_number(filename)
            if stim_num is None:
                logger.error(f"Could not extract stimulus number from {filename}")
                error_files += 1
                continue
            
            # Get the target directory name
            if stim_num not in target_dirs:
                logger.error(f"No matching target directory for stimulus number {stim_num} in {filename}")
                error_files += 1
                continue
                
            target_subdir = target_dirs[stim_num]
            target_path = os.path.join(target_dir, 'NATURAL', target_subdir)
            
            # Add duration suffix to the filename
            new_filename = add_duration_suffix(filename)
            
            if not dry_run:
                # Copy the file to the new location with the new filename
                target_file = os.path.join(target_path, new_filename)
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied {filename} to {target_file}")
            else:
                logger.info(f"Would copy {filename} to {os.path.join(target_path, new_filename)}")
            processed_files += 1
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            error_files += 1
    
    # Print summary
    logger.info("\nReorganization Summary:")
    logger.info(f"Total WAV files found: {total_files}")
    logger.info(f"Files processed: {processed_files}")
    logger.info(f"Files skipped: {skipped_files}")
    logger.info(f"Files with errors: {error_files}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Reorganize 10-second dataset structure')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--reference', default='2SECS', choices=['2SECS', '3SECS', '4SECS'],
                      help='Reference duration folder to use for structure (default: 2SECS)')
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Define the paths
    base_dir = "/om2/user/dlatorre/FORKED-REPO-METAMERS/model_metamers_pytorch/MMS_Scripts/Datasets_MMS"
    source_dir = os.path.join(base_dir, "10SECS_Norman-Haignere_McDermott_2018_Stimuli/Norman-Haignere_McDermott_2018_Stimuli2")
    target_dir = os.path.join(base_dir, "10SECS_Norman-Haignere_McDermott_2018_Stimuli")
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    try:
        # Get reference structure
        reference_subdirs = get_reference_structure(base_dir, args.reference)
        
        # Create a backup of the original directory
        if not args.dry_run:
            backup_dir = f"{target_dir}_backup"
            if not os.path.exists(backup_dir):
                shutil.copytree(target_dir, backup_dir)
                logger.info(f"Created backup at {backup_dir}")
        
        # Reorganize the files
        reorganize_files(source_dir, target_dir, reference_subdirs, args.dry_run)
        logger.info("Reorganization complete!")
        
    except Exception as e:
        logger.error(f"Error during reorganization: {e}")
        raise

if __name__ == "__main__":
    main() 