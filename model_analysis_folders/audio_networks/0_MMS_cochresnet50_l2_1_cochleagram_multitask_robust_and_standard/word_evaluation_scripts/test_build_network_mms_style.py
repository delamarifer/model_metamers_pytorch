#!/usr/bin/env python3
"""
Test script for build_network.py using the same pattern as make_mms.py
This script tests that the model loaded by build_network.py performs well on word classification
before proceeding with metamer generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os
import time
import argparse

# Add the parent directory to path to import build_network
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_network import build_net
from analysis_scripts.input_helpers import generate_import_audio_functions

def preproc_sound_np(sound):
    '''
    Sounds going into the pytorch models are normalized to have rms=0.1
    Additional preprocessing happens inside of the model loop. 
    '''
    sound = sound - np.mean(sound)
    sound = sound/np.sqrt(np.mean(sound**2))*0.1
    sound = np.expand_dims(sound, 0)
    sound = torch.from_numpy(sound).float().cuda()
    return sound

def calc_loss(model, inp, target, custom_loss, should_preproc=True):
    '''
    Modified from the Attacker module of Robustness. 
    Calculates the loss of an input with respect to target
    Uses custom loss (if provided) otherwise the criterion
    '''
    if should_preproc:
        inp = model.preproc(inp)
    return custom_loss(model.model, inp, target)

def test_model_performance_mms_style(model_type='robust', duration=2, num_test_samples=50):
    """
    Test model performance using the same pattern as make_mms.py
    """
    print(f"\n=== Testing Model Performance (MMS Style) ===")
    print(f"Model Type: {model_type}")
    print(f"Duration: {duration}s")
    print(f"Test Samples: {num_test_samples}")
    print("=" * 50)
    
    try:
        # Load model and dataset using the same pattern as make_mms.py
        print("Loading model and dataset...")
        ds_kwargs = {'duration': duration}
        
        model, ds, metamer_layers = build_net(
            include_rep_in_model=True, 
            use_normalization_for_audio_rep=True,
            ds_kwargs=ds_kwargs,
            strict=True,
            model_type=model_type,
            return_metamer_layers=True
        )
        
        print("✓ Model and dataset loaded successfully!")
        print(f"✓ Dataset type: {type(ds)}")
        print(f"✓ Metamer layers: {metamer_layers}")
        
        # Send model to GPU (same as make_mms.py)
        model = model.cuda()
        model.eval()
        
        # Turn off dropout for evaluation (same as make_mms.py)
        try:
            model.disable_dropout_functions()
            print('✓ Turned off dropout functions for evaluation')
        except:
            print('⚠ Could not disable dropout functions')
        
        # Test with a few samples first (like make_mms.py does)
        print("\n--- Testing Forward Pass with Sample Data ---")
        
        # Load some test data using the same pattern as make_mms.py
        INPUTAUDIOFUNCNAME = 'natural_sounds_norman_haignere'
        INPUTAUDIOFUNC = generate_import_audio_functions(INPUTAUDIOFUNCNAME, rms_normalize=0.1)
        
        if INPUTAUDIOFUNC is None:
            raise ValueError(f"Could not import audio function: {INPUTAUDIOFUNCNAME}")
        
        # Test with speech samples instead of environmental sounds
        # Use indices that correspond to speech clips: man_speaking and woman_speaking
        # Note: The natural_sounds_norman_haignere function always labels everything as "about"
        # regardless of the actual audio content, so we're testing if the model can distinguish
        # speech from non-speech rather than specific words
        speech_test_indices = [232, 414, 508, 506, 504, 503, 502, 501]  # Speech samples
        non_speech_test_indices = [102, 11, 150, 211, 224]  # Non-speech: crumpling_paper, baby_crying, finger_tapping, keys_jingling, walking_on_leaves
        all_test_indices = speech_test_indices + non_speech_test_indices
        correct_predictions = 0
        total_predictions = 0
        
        print(f"Testing {len(all_test_indices)} samples ({len(speech_test_indices)} speech, {len(non_speech_test_indices)} non-speech)...")
        print("Note: All samples are labeled as 'about' by the audio function, but contain different content.")
        print("We're testing if the model can distinguish speech from non-speech.")
        
        for SIDX in all_test_indices:
            try:
                # Load audio using the same pattern as make_mms.py
                audio_dict = INPUTAUDIOFUNC(SIDX, SR=ds.SR, duration=duration)
                audio_dict['wav_orig'] = audio_dict['wav'].copy()
                
                # Transform audio using dataset transform (same as make_mms.py)
                audio_dict['wav'], _ = ds.transform_test(audio_dict['wav_orig'], None)
                
                # Prepare input (same as make_mms.py)
                im = audio_dict['wav'].float()
                label_keys = ds.label_mapping.keys()
                label_values = ds.label_mapping.values()
                label_idx = list(label_keys)[list(label_values).index(audio_dict['correct_response'])]
                targ = torch.from_numpy(np.array([label_idx])).float()
                
                # Run forward pass (same as make_mms.py)
                with torch.no_grad():
                    (predictions, rep, all_outputs), orig_im = model(im.cuda(), with_latent=True, fake_relu=True)
                
                # Extract predictions (same as make_mms.py)
                orig_predictions = []
                for b in range(1):  # BATCH_SIZE=1
                    try:
                        orig_predictions.append(predictions[b].detach().cpu().numpy())
                    except KeyError:
                        orig_predictions.append(predictions['signal/word_int'][b].detach().cpu().numpy())
                
                # Get predicted category and confidence
                pred_probs = orig_predictions[0].ravel()
                pred_idx = np.argmax(pred_probs)
                pred_confidence = pred_probs[pred_idx]
                orig_cat_prediction = [ds.label_mapping[pred_idx] for p in orig_predictions]
                true_category = audio_dict['correct_response']
                
                # Get top 3 predictions for better debugging
                top_3_indices = np.argsort(pred_probs)[-3:][::-1]
                top_3_predictions = [(ds.label_mapping[idx], pred_probs[idx]) for idx in top_3_indices]
                
                # Print detailed results
                sample_type = "SPEECH" if SIDX in speech_test_indices else "NON-SPEECH"
                print(f"\nSample {SIDX} ({sample_type}) ({audio_dict.get('stimulus_name', 'unknown')}):")
                print(f"  True Label: '{true_category}'")
                print(f"  Predicted: '{orig_cat_prediction[0]}' (confidence: {pred_confidence:.3f})")
                print(f"  Top 3 predictions:")
                for i, (pred, conf) in enumerate(top_3_predictions):
                    marker = "✓" if pred == true_category else " "
                    print(f"    {i+1}. {marker} '{pred}' (confidence: {conf:.3f})")
                
                if orig_cat_prediction[0] == true_category:
                    correct_predictions += 1
                    print(f"  ✓ CORRECT!")
                else:
                    print(f"  ✗ WRONG")
                total_predictions += 1
                
            except Exception as e:
                print(f"Error processing sample {SIDX}: {str(e)}")
                continue
        
        sample_accuracy = 100 * correct_predictions / total_predictions
        print(f"\n✓ Sample Test Accuracy: {sample_accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        
        # Now test with more samples using the dataset's validation loader
        print(f"\n--- Testing with Dataset Validation Loader ({num_test_samples} samples) ---")
        
        # Create validation dataloader
        _, val_loader = ds.make_loaders(workers=4, 
                                       batch_size=32,
                                       data_aug=False,  # No augmentation for evaluation
                                       only_val=True)  # Only get validation loader
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", total=min(num_test_samples//32 + 1, len(val_loader))):
                if batch_count * 32 >= num_test_samples:
                    break
                    
                inputs = batch[0].cuda()
                word_labels = batch[1]['signal/word_int'].cuda()
                
                # Get model outputs (same pattern as make_mms.py)
                outputs = model(inputs)
                
                # Extract word predictions from the output tuple
                if isinstance(outputs, tuple):
                    predictions = outputs[0]  # First element is predictions
                    if isinstance(predictions, dict):
                        word_predictions = predictions['signal/word_int']
                    else:
                        word_predictions = predictions
                else:
                    word_predictions = outputs
                
                _, predicted = torch.max(word_predictions, 1)
                total += word_labels.size(0)
                correct += (predicted == word_labels).sum().item()
                
                batch_count += 1
        
        accuracy = 100 * correct / total
        print(f"✓ Validation Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Overall assessment
        print(f"\n=== Performance Assessment ===")
        print(f"Sample Test Accuracy: {sample_accuracy:.2f}%")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        if accuracy >= 80 and sample_accuracy >= 80:
            print("✓ EXCELLENT: Model performance is ≥80% - Ready for metamer generation!")
            return True, accuracy, sample_accuracy
        elif accuracy >= 60 and sample_accuracy >= 60:
            print("⚠ ACCEPTABLE: Model performance is ≥60% - Proceed with caution.")
            return True, accuracy, sample_accuracy
        else:
            print("✗ POOR: Model performance is <60% - Check model before proceeding.")
            return False, accuracy, sample_accuracy
            
    except Exception as e:
        print(f"✗ Model performance test failed: {str(e)}")
        raise

def test_different_configurations():
    """
    Test the model with different configurations (like make_mms.py supports)
    """
    print("\n=== Testing Different Configurations ===")
    
    # Test different model types and durations (same as make_mms.py supports)
    model_types = ['robust', 'standard']
    durations = [2, 3, 4, 7, 10]
    
    results = {}
    
    for model_type in model_types:
        results[model_type] = {}
        
        for duration in durations:
            try:
                print(f"\n--- Testing {model_type} model with {duration}s duration ---")
                
                success, val_accuracy, sample_accuracy = test_model_performance_mms_style(
                    model_type=model_type, 
                    duration=duration, 
                    num_test_samples=50
                )
                
                results[model_type][duration] = {
                    'success': success,
                    'val_accuracy': val_accuracy,
                    'sample_accuracy': sample_accuracy,
                    'status': 'success'
                }
                
                print(f"✓ Configuration test passed!")
                
            except Exception as e:
                print(f"✗ Configuration test failed: {str(e)}")
                results[model_type][duration] = {
                    'success': False,
                    'val_accuracy': None,
                    'sample_accuracy': None,
                    'status': 'failed',
                    'error': str(e)
                }
    
    return results

def main():
    """
    Main test function
    """
    print("=== Testing build_network.py (MMS Style) ===")
    print("This script tests that your build_network.py loads models correctly")
    print("and performs well on word classification tasks using the same pattern as make_mms.py.")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test basic functionality first
    try:
        print("\n--- Basic Functionality Test ---")
        success, val_accuracy, sample_accuracy = test_model_performance_mms_style(
            model_type='robust', 
            duration=2, 
            num_test_samples=100
        )
        
        if success:
            print("\n✓ Basic functionality test passed!")
            print("✓ Your build_network.py is working correctly for metamer generation!")
        else:
            print("\n✗ Basic functionality test failed!")
            print("✗ Please check build_network.py before proceeding with metamer generation.")
            return
            
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {str(e)}")
        print("build_network.py needs to be fixed before proceeding.")
        return
    
    # Test different configurations
    print("\n--- Testing Different Configurations ---")
    results = test_different_configurations()
    
    # Print summary
    print("\n=== Test Summary ===")
    for model_type, durations in results.items():
        print(f"\n{model_type.upper()} Model:")
        for duration, result in durations.items():
            status = result['status']
            if status == 'success':
                val_acc = result['val_accuracy']
                sample_acc = result['sample_accuracy']
                print(f"  {duration}s: ✓ Val={val_acc:.1f}%, Sample={sample_acc:.1f}%")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  {duration}s: ✗ Failed - {error}")
    
    print("\n=== Test Complete ===")
    print("If all tests passed, your build_network.py is ready for metamer generation!")
    print("You can now run make_mms.py with confidence.")

if __name__ == "__main__":
    main() 