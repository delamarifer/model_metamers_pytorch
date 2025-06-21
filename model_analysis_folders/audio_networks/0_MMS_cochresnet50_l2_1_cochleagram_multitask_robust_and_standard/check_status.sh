#!/bin/bash

# Simple wrapper script to check completion status
# Usage: ./check_status.sh <run_number>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <run_number>"
    echo "Example: $0 41814329"
    exit 1
fi

RUN_NUMBER=$1

echo "Checking completion status for run $RUN_NUMBER..."
echo "================================================"

python check_completion_status.py $RUN_NUMBER

echo ""
echo "To check specific sound IDs, model types, or subclips, use:"
echo "python check_completion_status.py $RUN_NUMBER --sound-ids 5 6 7 18"
echo "python check_completion_status.py $RUN_NUMBER --model-types standard"
echo "python check_completion_status.py $RUN_NUMBER --subclips 1 2"
echo "python check_completion_status.py $RUN_NUMBER --random-seeds 42 123" 