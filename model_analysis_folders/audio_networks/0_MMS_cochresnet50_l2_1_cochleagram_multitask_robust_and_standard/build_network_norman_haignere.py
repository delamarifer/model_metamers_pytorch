import sys
from robustness.datasets import NormanHaignereMcDermott
from robustness.model_utils import make_and_restore_model

from model_analysis_folders.all_model_info import MODEL_BASE_PATH
import os

def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True,
              model_type='robust',
              duration=3):  # Duration parameter for Norman-Haignere McDermott

    # Build the dataset
    ds = NormanHaignereMcDermott(MODEL_BASE_PATH, 
                                include_rep_in_model=include_rep_in_model, 
                                audio_representation='cochleagram_1',
                                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                                include_identity_sequential=include_identity_sequential,
                                duration=duration,
                                **ds_kwargs)

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
        'input_after_preproc',
        'conv1',
        'bn1',
        'conv1_relu1',
        'maxpool1',
        'layer1',
        'layer2',
        'layer3',
        'layer4',
        'avgpool',
        'final'
    ]

    # Path to the network checkpoint to load
    if model_type == 'robust':
        resume_path = '/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/fmri_comparison_networks/for_component_tests/cochresnet50_l2_1_robust_cochleagram_multitask_increase_audioset_weight/standard_training_word_and_audioset_and_speaker_decay_lr_l2_1_robust_training_increase_audioset_weight.pt'
    else:  # standard
        resume_path = '/rdma/vast-rdma/vast/mcdermott/dlatorre/STAND/cochdnn/model_checkpoints/audio_rep_training_cochleagram_1/standard_training_word_and_audioset_and_speaker_decay_lr/542752d7-9849-49ff-b84a-6758a81585b4/5_checkpoint.pt'

    # Restore the model
    print(f"ds_kwargs in build_net: {ds_kwargs}")

    model, _ = make_and_restore_model(arch='resnet50', 
                                    dataset=ds, 
                                    parallel=False,
                                    resume_path=resume_path,
                                    strict=strict,
                                    arch_kwargs=ds_kwargs,
                                    model_type=model_type)

    # send the model to the GPU and return it. 
    model.cuda()
    model.eval()

    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(include_rep_in_model=True,
         use_normalization_for_audio_rep=True,
         return_metamer_layers=False,
         include_identity_sequential=False,
         ds_kwargs={},
         model_type='robust',
         strict=True,
         duration=3):  # Duration parameter
    # This parameter is not used for this model
    print(f"ds_kwargs inside build_network.main: {ds_kwargs}")
    print(f"Using model type: {model_type}")
    print(f"Using duration: {duration} seconds")

    if return_metamer_layers:
        model, ds, metamer_layers = build_net(include_rep_in_model=include_rep_in_model,
                                            use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                                            return_metamer_layers=return_metamer_layers,
                                            strict=strict,
                                            include_identity_sequential=include_identity_sequential,
                                            ds_kwargs=ds_kwargs,
                                            model_type=model_type,
                                            duration=duration)
        return model, ds, metamer_layers
    else:
        model, ds = build_net(include_rep_in_model=include_rep_in_model,
                            use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                            return_metamer_layers=return_metamer_layers,
                            strict=strict,
                            include_identity_sequential=include_identity_sequential,
                            ds_kwargs=ds_kwargs,
                            model_type=model_type,
                            duration=duration)
        return model, ds

if __name__== "__main__":
    main() 