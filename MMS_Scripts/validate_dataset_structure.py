import os
import logging
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define expected segment suffixes for each duration
EXPECTED_SEGMENTS = {
    '2': ['00_02', '02_04', '04_06', '06_08', '08_10'],
    '3': ['01_04', '04_07', '07_10'],
    '4': ['01_05', '05_09'],
    '10': ['00_10'],
}

def get_duration_folders(base_dir):
    """Get all duration folders in the base directory."""
    return [d for d in os.listdir(base_dir) 
            if d.endswith('SECS_Norman-Haignere_McDermott_2018_Stimuli')]

def get_folder_structure(duration_folder):
    """Get the folder structure for a given duration dataset."""
    natural_path = os.path.join(duration_folder, 'NATURAL')
    if not os.path.exists(natural_path):
        return []
    
    # Get all subdirectories and sort them
    subdirs = [d for d in os.listdir(natural_path) 
              if os.path.isdir(os.path.join(natural_path, d))]
    return sorted(subdirs)

def get_file_segments(duration_folder):
    """Get the segment suffixes for each file in each folder."""
    natural_path = os.path.join(duration_folder, 'NATURAL')
    if not os.path.exists(natural_path):
        return {}
    
    segments = defaultdict(lambda: defaultdict(list))
    for root, _, files in os.walk(natural_path):
        for file in files:
            if file.endswith('.wav'):
                folder = os.path.basename(root)
                # Extract segment suffix (e.g., 00_02, 02_04, etc.)
                parts = file.split('_')
                if len(parts) >= 2:
                    segment = '_'.join(parts[-2:]).replace('.wav', '')
                    segments[folder][segment].append(file)
    return segments

def validate_structures(base_dir):
    """Validate the consistency of folder and file structures across all duration datasets."""
    # Get all duration folders
    duration_folders = get_duration_folders(base_dir)
    logger.info("Found duration folders: %s", duration_folders)
    
    # Get folder structures for each duration
    folder_structures = {}
    for duration in duration_folders:
        full_path = os.path.join(base_dir, duration)
        folder_structures[duration] = get_folder_structure(full_path)
    
    # Compare folder structures
    reference = folder_structures[duration_folders[0]]
    logger.info("\nUsing %s as reference structure", duration_folders[0])
    
    for duration, structure in folder_structures.items():
        if structure != reference:
            logger.warning("\n%s has different folder structure:", duration)
            missing = set(reference) - set(structure)
            extra = set(structure) - set(reference)
            if missing:
                logger.warning("Missing folders: %s", sorted(missing))
            if extra:
                logger.warning("Extra folders: %s", sorted(extra))
        else:
            logger.info("%s folder structure matches reference", duration)
    
    # Validate file segment patterns
    logger.info("\nValidating file segment patterns...")
    for duration in duration_folders:
        full_path = os.path.join(base_dir, duration)
        # Extract the expected segment list for this duration
        dur_key = duration.split('SECS')[0]
        dur_key = dur_key.lstrip('0')  # Remove leading zeros if any
        expected_segments = EXPECTED_SEGMENTS.get(dur_key, [])
        segments = get_file_segments(full_path)
        
        missing_segments = defaultdict(list)
        extra_segments = defaultdict(list)
        for folder in folder_structures[duration]:
            found_segments = set(segments[folder].keys())
            expected_set = set(expected_segments)
            missing = expected_set - found_segments
            extra = found_segments - expected_set
            if missing:
                missing_segments[folder] = sorted(missing)
            if extra:
                extra_segments[folder] = sorted(extra)
        
        if missing_segments:
            logger.warning("\n%s is missing expected segments:", duration)
            for folder, segs in missing_segments.items():
                logger.warning("  %s: missing %s", folder, segs)
        if extra_segments:
            logger.warning("\n%s has extra segments:", duration)
            for folder, segs in extra_segments.items():
                logger.warning("  %s: extra %s", folder, segs)
        if not missing_segments and not extra_segments:
            logger.info("%s segment structure is correct", duration)

def main():
    # Define the base directory
    base_dir = "/om2/user/dlatorre/FORKED-REPO-METAMERS/model_metamers_pytorch/MMS_Scripts/Datasets_MMS/MMS_Datasets_Norman-Haignere_McDermott_2018"
    
    try:
        validate_structures(base_dir)
    except Exception as e:
        logger.error("Error during validation: %s", e)
        raise

if __name__ == "__main__":
    main() 