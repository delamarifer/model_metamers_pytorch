import os
import subprocess
import sys
import h5py
import numpy as np
from robustness.tools.audio_helpers import load_audio_wav_resample
from analysis_scripts.default_paths import fMRI_DATA_PATH
import torch
import shutil
import time
from datetime import datetime, timedelta

# Add parent directory to path to import build_network
sys.path.append('..')
import build_network

# Settings
MODEL_TYPES = ['robust', 'standard']
NUM_SPLITS = 10
RANDSEED = 3882
OVERWRITE = True
OVERWRITE_FEATURES = True
FEATURES_DIR = './features'  # Directory to save/load features (local to subfolder)
PLOTS_DIR = './plots_fmri_predictivity_analysis'  # Remove timestamp for consistency
REGRESSION_SCRIPT = '../../../../analysis_scripts/run_regressions_all_voxels_om_natsounddata.py'

# Delete and recreate features and plots directories
for d in [FEATURES_DIR, PLOTS_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

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

for model_type in MODEL_TYPES:
    print(f'Processing model: {model_type}')
    start_time = time.time()
    
    # Get model, dataset, and layers
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True, model_type=model_type)
    print_model_info(model_type, model, metamer_layers)
    # Remove commented-out layers if any
    metamer_layers = [l for l in metamer_layers if not l.startswith('#')]

    # 1. Feature extraction (all layers at once, if not already done)
    feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
    if OVERWRITE_FEATURES or not os.path.exists(feature_file):
        print(f'    Extracting features for all layers and saving to {feature_file}')
        feature_start_time = time.time()
        
        # Load audio metadata
        sound_list = np.load(os.path.join(fMRI_DATA_PATH, 'neural_stim_meta.npy'))
        wavs_location = os.path.join(fMRI_DATA_PATH, '165_natural_sounds')
        SR = 20000
        MEASURE_DUR = 2
        wav_array = np.empty([165, SR*MEASURE_DUR])
        
        print(f'    Loading {len(sound_list)} audio files...')
        audio_load_start = time.time()
        for wav_idx, wav_data in enumerate(sound_list):
            test_audio, _ = load_audio_wav_resample(os.path.join(wavs_location, wav_data[0].decode('utf-8')), DUR_SECS=MEASURE_DUR, resample_SR=SR)
            wav_array[wav_idx,:] = test_audio/np.sqrt(np.mean(test_audio**2))
        audio_load_time = time.time() - audio_load_start
        print(f'    Audio loading completed in {audio_load_time:.1f}s')
        
        # Prepare HDF5 file
        with h5py.File(feature_file, 'w') as h5f:
            h5f.create_dataset('layer_list', data=np.array([l.encode('utf-8') for l in metamer_layers]))
            dsets = {}
            # Run model for each sound, collect all layer activations
            print(f'    Processing {len(sound_list)} sounds through {len(metamer_layers)} layers...')
            model_start_time = time.time()
            for sound_idx, sound_info in enumerate(sound_list):
                if sound_idx % 10 == 0:  # Progress update every 10 sounds
                    elapsed = time.time() - model_start_time
                    if sound_idx > 0:
                        avg_time_per_sound = elapsed / sound_idx
                        remaining_sounds = len(sound_list) - sound_idx
                        eta = avg_time_per_sound * remaining_sounds
                        eta_str = str(timedelta(seconds=int(eta)))
                        print(f'      Progress: {sound_idx}/{len(sound_list)} sounds ({sound_idx/len(sound_list)*100:.1f}%) - ETA: {eta_str}')
                
                sound = preproc_sound_np(wav_array[sound_idx,:])
                with torch.no_grad():
                    model_out = model(sound, with_latent=True)
                    _, _, layer_returns = model_out[0]
                # Only print debug info for the first sound
                if sound_idx == 0:
                    print(f"layer_returns type: {type(layer_returns)}")
                    if isinstance(layer_returns, dict):
                        print(f"Available keys: {list(layer_returns.keys())}")
                    else:
                        print(f"layer_returns shape: {getattr(layer_returns, 'shape', 'N/A')}")
                for layer_idx, layer_name in enumerate(metamer_layers):
                    if isinstance(layer_returns, dict):
                        activ = layer_returns[layer_name]
                        # Only print debug info for the first sound and first layer
                        if sound_idx == 0 and layer_idx == 0:
                            print(f"Layer: {layer_name}, activ shape: {activ.shape}, activ ndim: {activ.ndim}")
                        activ_np = activ.cpu().detach().numpy()
                        # Time-average if 4D (NCHW)
                        if activ_np.ndim == 4:
                            activ_np = np.mean(activ_np, 3).ravel()
                        else:
                            activ_np = activ_np.ravel()
                        if sound_idx == 0:
                            feature_dim = activ_np.shape[0]
                            print(f"Creating dataset for {layer_name} with shape (165, {feature_dim})")
                            dsets[layer_name] = h5f.create_dataset(layer_name, (165, feature_dim), dtype='float32')
                        dsets[layer_name][sound_idx, :] = activ_np
                    else:
                        if sound_idx == 0 and layer_idx == 0:
                            print(f"Cannot extract layer {layer_name}: layer_returns is not a dict.")
        
        feature_time = time.time() - feature_start_time
        print(f'    Feature extraction completed in {feature_time:.1f}s ({feature_time/60:.1f} minutes)')
        print(f'    Saved features for all layers to {feature_file}')

    # 2. Run regression for each layer
    regression_dir = os.path.join('..', 'regression_results', f'natsound_activations_{model_type}')
    os.makedirs(regression_dir, exist_ok=True)
    
    print(f'    Running regression for {len(metamer_layers)} layers...')
    regression_start_time = time.time()
    for layer_idx, layer_name in enumerate(metamer_layers):
        layer_start_time = time.time()
        print(f'  Layer {layer_idx + 1}/{len(metamer_layers)}: {layer_name}')
        print(f'    Running regression for {layer_name}...')
        cmd = [
            'python', REGRESSION_SCRIPT,
            str(layer_idx), feature_file,
            str(NUM_SPLITS),
            str(RANDSEED),
            str(OVERWRITE)
        ]
        print('    Command:', ' '.join(map(str, cmd)))
        subprocess.run(cmd)
        
        layer_time = time.time() - layer_start_time
        print(f'    Layer {layer_name} completed in {layer_time:.1f}s')
        
        # Estimate remaining time
        if layer_idx < len(metamer_layers) - 1:
            avg_time_per_layer = (time.time() - regression_start_time) / (layer_idx + 1)
            remaining_layers = len(metamer_layers) - layer_idx - 1
            eta = avg_time_per_layer * remaining_layers
            eta_str = str(timedelta(seconds=int(eta)))
            print(f'    Estimated time remaining: {eta_str}')
    
    regression_time = time.time() - regression_start_time
    total_time = time.time() - start_time
    
    print(f'\nModel {model_type} completed:')
    print(f'  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)')
    if OVERWRITE_FEATURES or not os.path.exists(feature_file):
        print(f'  Feature extraction: {feature_time:.1f}s')
    print(f'  Regression: {regression_time:.1f}s')
    print(f'  Average time per layer: {regression_time/len(metamer_layers):.1f}s')

print(f'\nAll models completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}') 