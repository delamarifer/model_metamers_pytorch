#!/bin/bash

# Script to merge all metamer pickle files from different runs
# into a single folder with consistent naming

# Set the base directory (parent directory, since we're in mms_plotting subdirectory)
BASE_DIR="$(dirname "$(pwd)")"

# Set the output directory (in the current mms_plotting directory)
OUTPUT_DIR="merged_metamers"

echo "=== Metamer Merge Script ==="
echo "Base directory: $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "============================"

# First, run a dry run to see what would be done
echo "Running dry run to preview changes..."
python merge_metamer_pickles.py --base_dir "$BASE_DIR" --output_dir "$OUTPUT_DIR" --dry_run

echo ""
echo "Do you want to proceed with the actual merge? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Proceeding with merge..."
    python merge_metamer_pickles.py --base_dir "$BASE_DIR" --output_dir "$OUTPUT_DIR"
    echo "Merge complete!"
else
    echo "Merge cancelled."
fi 