import os
import sys
import h5py
import numpy as np
import time
from robustness.tools.audio_helpers import load_audio_wav_resample
from analysis_scripts.default_paths import fMRI_DATA_PATH
import torch
import shutil

# Add parent directory to path to import build_network
sys.path.append('..')
import build_network

# Settings
MODEL_TYPES = ['robust', 'standard']
OVERWRITE_FEATURES = True
FEATURES_DIR = './features'  # Directory to save/load features (local to subfolder)

# Delete and recreate features directory
if os.path.exists(FEATURES_DIR):
    shutil.rmtree(FEATURES_DIR)
os.makedirs(FEATURES_DIR, exist_ok=True)

def print_model_info(model_type, model, metamer_layers):
    print(f'\nModel type: {model_type}')
    # Print checkpoint path (from build_network.py logic)
    if model_type == 'robust':
        checkpoint = '/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/fmri_comparison_networks/for_component_tests/cochresnet50_l2_1_robust_cochleagram_multitask_increase_audioset_weight/standard_training_word_and_audioset_and_speaker_decay_lr_l2_1_robust_training_increase_audioset_weight.pt'
    else:
        checkpoint = '/rdma/vast-rdma/vast/mcdermott/dlatorre/STAND/cochdnn/model_checkpoints/audio_rep_training_cochleagram_1/standard_training_word_and_audioset_and_speaker_decay_lr/542752d7-9849-49ff-b84a-6758a81585b4/5_checkpoint.pt'
    print(f'Checkpoint: {checkpoint}')
    print('Metamer layers:')
    for idx, layer in enumerate(metamer_layers):
        print(f'  {idx}: {layer}')

def preproc_sound_np(sound):
    sound = sound - np.mean(sound)
    sound = sound/np.sqrt(np.mean(sound**2))*0.1
    sound = np.expand_dims(sound, 0)
    sound = torch.from_numpy(sound).float().cuda()
    return sound

def estimate_time(num_sounds, num_layers, model_types):
    """Estimate time for feature extraction"""
    # Rough estimates based on typical processing times
    time_per_sound_per_layer = 0.1  # seconds
    total_operations = num_sounds * num_layers * len(model_types)
    estimated_seconds = total_operations * time_per_sound_per_layer
    
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    seconds = int(estimated_seconds % 60)
    
    return f"{hours}h {minutes}m {seconds}s"

for model_type in MODEL_TYPES:
    print(f'Processing model: {model_type}')
    # Get model, dataset, and layers
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True, model_type=model_type)
    print_model_info(model_type, model, metamer_layers)
    # Remove commented-out layers if any
    metamer_layers = [l for l in metamer_layers if not l.startswith('#')]

    # Feature extraction (all layers at once)
    feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
    if OVERWRITE_FEATURES or not os.path.exists(feature_file):
        print(f'    Extracting features for all layers and saving to {feature_file}')
        
        # Load audio metadata
        sound_list = np.load(os.path.join(fMRI_DATA_PATH, 'neural_stim_meta.npy'))
        wavs_location = os.path.join(fMRI_DATA_PATH, '165_natural_sounds')
        SR = 20000
        MEASURE_DUR = 2
        wav_array = np.empty([165, SR*MEASURE_DUR])
        
        print(f"    Loading {len(sound_list)} audio files...")
        for wav_idx, wav_data in enumerate(sound_list):
            test_audio, _ = load_audio_wav_resample(os.path.join(wavs_location, wav_data[0].decode('utf-8')), DUR_SECS=MEASURE_DUR, resample_SR=SR)
            wav_array[wav_idx,:] = test_audio/np.sqrt(np.mean(test_audio**2))
        
        # Estimate time
        estimated_time_str = estimate_time(len(sound_list), len(metamer_layers), MODEL_TYPES)
        print(f"    Estimated time for feature extraction: {estimated_time_str}")
        
        # Prepare HDF5 file
        start_time = time.time()
        with h5py.File(feature_file, 'w') as h5f:
            h5f.create_dataset('layer_list', data=np.array([l.encode('utf-8') for l in metamer_layers]))
            dsets = {}
            
            # Run model for each sound, collect all layer activations
            for sound_idx, sound_info in enumerate(sound_list):
                if sound_idx % 10 == 0:  # Progress update every 10 sounds
                    elapsed = time.time() - start_time
                    remaining_sounds = len(sound_list) - sound_idx
                    if sound_idx > 0:
                        avg_time_per_sound = elapsed / sound_idx
                        estimated_remaining = remaining_sounds * avg_time_per_sound
                        print(f"    Progress: {sound_idx}/{len(sound_list)} sounds processed. "
                              f"Elapsed: {elapsed/60:.1f}m, "
                              f"Estimated remaining: {estimated_remaining/60:.1f}m")
                
                sound = preproc_sound_np(wav_array[sound_idx,:])
                with torch.no_grad():
                    model_out = model(sound, with_latent=True)
                    _, _, layer_returns = model_out[0]
                
                # Only print debug info for the first sound
                if sound_idx == 0:
                    print(f"    layer_returns type: {type(layer_returns)}")
                    if isinstance(layer_returns, dict):
                        print(f"    Available keys: {list(layer_returns.keys())}")
                    else:
                        print(f"    layer_returns shape: {getattr(layer_returns, 'shape', 'N/A')}")
                
                for layer_idx, layer_name in enumerate(metamer_layers):
                    if isinstance(layer_returns, dict):
                        activ = layer_returns[layer_name]
                        # Only print debug info for the first sound and first layer
                        if sound_idx == 0 and layer_idx == 0:
                            print(f"    Layer: {layer_name}, activ shape: {activ.shape}, activ ndim: {activ.ndim}")
                        activ_np = activ.cpu().detach().numpy()
                        # Time-average if 4D (NCHW)
                        if activ_np.ndim == 4:
                            activ_np = np.mean(activ_np, 3).ravel()
                        else:
                            activ_np = activ_np.ravel()
                        if sound_idx == 0:
                            feature_dim = activ_np.shape[0]
                            print(f"    Creating dataset for {layer_name} with shape (165, {feature_dim})")
                            dsets[layer_name] = h5f.create_dataset(layer_name, (165, feature_dim), dtype='float32')
                        dsets[layer_name][sound_idx, :] = activ_np
                    else:
                        if sound_idx == 0 and layer_idx == 0:
                            print(f"    Cannot extract layer {layer_name}: layer_returns is not a dict.")
        
        total_time = time.time() - start_time
        print(f'    Saved features for all layers to {feature_file}')
        print(f'    Total feature extraction time: {total_time/60:.1f} minutes')
    else:
        print(f'    Features already exist at {feature_file}, skipping extraction')

print("\n=== Feature Extraction Complete ===")
print("You can now run individual layer regressions using:")
print("  python run_regression_single_layer.py <model_type> <layer_idx>")
print("  or")
print("  python run_regression_all_layers.py") 