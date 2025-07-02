import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the parent directory to path to import build_network
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_network import build_net
from analysis_scripts.input_helpers import generate_import_audio_functions

def evaluate_model(model, dataloader, device, ds):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch[0].to(device)
            word_labels = batch[1]['signal/word_int'].to(device)
            
            # Get model outputs
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
            
            # Print predictions for each sample in batch
            for i in range(word_labels.size(0)):
                true_label = ds.label_mapping[int(word_labels[i].cpu().numpy())]
                pred_label = ds.label_mapping[int(predicted[i].cpu().numpy())]
                print(f'True: {true_label}, Predicted: {pred_label}')
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    
    # Build model and dataset
    print("Loading model and dataset...")
    model, ds = build_net(include_rep_in_model=True, 
                         use_normalization_for_audio_rep=True,
                         strict=True)
    
    # Create evaluation dataloader using the dataset's make_loaders method
    _, eval_loader = ds.make_loaders(workers=4, 
                                   batch_size=32,
                                   data_aug=False,  # No augmentation for evaluation
                                   only_val=True)  # Only get validation loader
    
    # Evaluate model
    print("Starting evaluation...")
    accuracy = evaluate_model(model, eval_loader, device, ds)
    
    print("\nWord Classification Accuracy: {:.2f}%".format(accuracy))
    
    if accuracy >= 80:
        print("\nModel performance is above 80% - Proceeding with metamer generation is recommended.")
    else:
        print("\nModel performance is below 80% - Please check the model before proceeding with metamer generation.")

if __name__ == "__main__":
    main() 