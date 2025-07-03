#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path
from create_metamer_heatmaps import parse_filename

# Test with first pickle file
pickle_files = list(Path('merged_metamers').glob('*.pckl'))
print(f'Found {len(pickle_files)} pickle files')

if pickle_files:
    test_file = pickle_files[0]
    print(f'Testing with: {test_file.name}')
    
    # Test filename parsing
    parsed = parse_filename(test_file.name)
    print('Parsed filename:', parsed)
    
    # Test pickle loading
    try:
        with open(test_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        print('Pickle loaded successfully')
        print('Keys in data:', list(data.keys()) if isinstance(data, dict) else 'Not a dict')
        
        if isinstance(data, dict):
            orig_keys = list(data.get('all_outputs_orig', {}).keys())
            synth_keys = list(data.get('all_outputs_out_dict', {}).keys())
            print('Original layers:', orig_keys[:5], '...' if len(orig_keys) > 5 else '')
            print('Synthesized layers:', synth_keys[:5], '...' if len(synth_keys) > 5 else '')
            
    except Exception as e:
        print(f'Error loading pickle: {e}')
else:
    print('No pickle files found') 