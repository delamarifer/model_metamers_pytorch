import sys
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model
from robustness.audio_functions import jsinV3_loss_functions

from model_analysis_folders.all_model_info import JSIN_PATH, MODEL_BASE_PATH
import os

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True,
              model_type='robust'):

    # Build the dataset so that the number of classes and normalization 
    # is set appropriately. Not needed for metamer generation, but ds is 
    # used for eval scripts.  

    print(ds_kwargs,"DSKWA")
    
    # Determine which cochleagram configuration to use based on duration
    duration = ds_kwargs.get('duration', 2)  # Default to 2 seconds if not specified
    if duration == 2:
        audio_representation = 'cochleagram_1'
    elif duration == 3:
        audio_representation = 'cochleagram_1_3_secs'
    elif duration == 4:
        audio_representation = 'cochleagram_1_4_secs'
    elif duration == 7:
        audio_representation = 'cochleagram_1_7_secs'
    elif duration == 10:
        audio_representation = 'cochleagram_1_10_secs'
    else:
        raise ValueError(f"Unsupported duration: {duration}. Supported durations are 2, 3, 4, 7, and 10 seconds.")
    
    ds = jsinV3(JSIN_PATH, include_rep_in_model=include_rep_in_model, 
                audio_representation=audio_representation,
                include_all_labels=True,
                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                include_identity_sequential=include_identity_sequential, 
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    if model_type == 'robust':
        resume_path = '/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/fmri_comparison_networks/for_component_tests/cochresnet50_l2_1_robust_cochleagram_multitask_increase_audioset_weight/standard_training_word_and_audioset_and_speaker_decay_lr_l2_1_robust_training_increase_audioset_weight.pt'
    else:  # standard
        resume_path = '/rdma/vast-rdma/vast/mcdermott/dlatorre/STAND/cochdnn/model_checkpoints/audio_rep_training_cochleagram_1/standard_training_word_and_audioset_and_speaker_decay_lr/542752d7-9849-49ff-b84a-6758a81585b4/5_checkpoint.pt'

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
         'final/signal/word_int',
         'final/signal/speaker_int',
         'final/noise/labels_binary_via_int',
    ]

    TASK_LOSS_PARAMS={}
    TASK_LOSS_PARAMS['signal/word_int'] = {
        'loss_type': 'crossentropyloss',
        'weight': 1.0
    }
    TASK_LOSS_PARAMS['noise/labels_binary_via_int'] = {
        'loss_type': 'bcewithlogitsloss',
        'weight': 300.0
    }
    TASK_LOSS_PARAMS['signal/speaker_int'] = {
        'loss_type': 'crossentropyloss',
        'weight': 0.25
    }

    PLACEHOLDER_BATCH_SIZE=None
    custom_loss = jsinV3_loss_functions.jsinV3_multi_task_loss(TASK_LOSS_PARAMS, PLACEHOLDER_BATCH_SIZE).cuda()
    custom_adv_criterion = jsinV3_loss_functions.jsinV3_multi_task_loss(TASK_LOSS_PARAMS, PLACEHOLDER_BATCH_SIZE, reduction='none')

    def calc_custom_adv_loss_with_batch_size(model, inp, target, BATCH_SIZE):
        '''
        Wraps the adversarial criterion to take in the model.
        '''
        output = model(inp)
        custom_adv_criterion.set_batch_size(BATCH_SIZE)
        loss = custom_adv_criterion(output, target)
        return loss, output

    # Store the custom loss parameters within ds, so that we return it
    ds.multitask_parameters = {'TASK_LOSS_PARAMS': TASK_LOSS_PARAMS,
                               'custom_loss': custom_loss,
                               'custom_adv_criterion': custom_adv_criterion,
                               'calc_custom_adv_loss_with_batch_size': calc_custom_adv_loss_with_batch_size}

    # Restore the model
    print(f"ds_kwargs in build_net: {ds_kwargs}")

    model, _ = make_and_restore_model(arch='resnet_multi_task50', 
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
         strict=True):
    # This parameter is not used for this model
    print(f"ds_kwargs inside build_network.main: {ds_kwargs}")
    print(f"Using model type: {model_type}")

    if return_metamer_layers:
        model, ds, metamer_layers = build_net(include_rep_in_model=include_rep_in_model,
                                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                                              return_metamer_layers=return_metamer_layers,
                                              strict=strict,
                                              include_identity_sequential=include_identity_sequential,
                                              ds_kwargs=ds_kwargs,
                                              model_type=model_type)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(include_rep_in_model=include_rep_in_model,
                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                              return_metamer_layers=return_metamer_layers,
                              strict=strict,
                              include_identity_sequential=include_identity_sequential,
                              ds_kwargs=ds_kwargs,
                              model_type=model_type)
        return model, ds

if __name__== "__main__":
    main()