import pandas as pd
import numpy as np

# Read the CSV file
csv_file = "model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/mms_plotting/test_heatmaps_simplified/processed_metamer_data.csv"
df = pd.read_csv(csv_file)

# Get unique sound IDs
all_sound_ids = sorted(df['sound_id'].unique())
print(f"Total unique sound IDs in dataset: {len(all_sound_ids)}")
print(f"Sound IDs: {all_sound_ids}")

# Check for each layer
layers_to_check = ['conv1_relu1', 'layer1', 'layer2']

for layer in layers_to_check:
    print(f"\n=== Analysis for {layer} ===")
    
    # Get data for this layer
    layer_data = df[df['layer'] == layer]
    
    if len(layer_data) == 0:
        print(f"  No data found for layer '{layer}'")
        continue
    
    # Get sound IDs that have data for this layer
    sound_ids_with_data = sorted(layer_data['sound_id'].unique())
    
    # Find missing sound IDs
    missing_sound_ids = [sid for sid in all_sound_ids if sid not in sound_ids_with_data]
    
    print(f"  Sound IDs with data: {sound_ids_with_data}")
    print(f"  Number of sound IDs with data: {len(sound_ids_with_data)}")
    
    if missing_sound_ids:
        print(f"  Missing sound IDs: {missing_sound_ids}")
        print(f"  Number of missing sound IDs: {len(missing_sound_ids)}")
    else:
        print(f"  All sound IDs have data for {layer}")

# Also check what layers are actually present in the data
print(f"\n=== All layers present in dataset ===")
all_layers = sorted(df['layer'].unique())
print(f"Layers: {all_layers}")
print(f"Total number of layers: {len(all_layers)}")

# Check data completeness for each sound ID across all layers
print(f"\n=== Data completeness by sound ID ===")
for sound_id in all_sound_ids:
    sound_data = df[df['sound_id'] == sound_id]
    layers_for_sound = sound_data['layer'].unique()
    missing_layers = [layer for layer in layers_to_check if layer not in layers_for_sound]
    
    if missing_layers:
        print(f"Sound ID {sound_id}: Missing layers {missing_layers}")
    else:
        print(f"Sound ID {sound_id}: Complete data for all requested layers") 