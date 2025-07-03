import sys
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model

from model_analysis_folders.all_model_info import JSIN_PATH, MODEL_BASE_PATH
import os

# Make a custom build script for spectemp_filts_time_average_coch1
def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True):

    # Build the dataset so that the number of classes and normalization 
    # is set appropriately. Not needed for metamer generation, but ds is 
    # used for eval scripts.  

    print(ds_kwargs,"DSKWA")
    
    # Map durations to their corresponding cochleagram configurations and dimensions
    duration_to_config = {
        2: ('cochleagram_1', (211, 390)),      # 2 seconds: 211 freq, 390 time bins
        3: ('cochleagram_1_3_secs', (211, 585)),  # 3 seconds: 211 freq, 585 time bins  
        4: ('cochleagram_1_4_secs', (211, 780)),  # 4 seconds: 211 freq, 780 time bins
        7: ('cochleagram_1_7_secs', (211, 1365)), # 7 seconds: 211 freq, 1365 time bins
        10: ('cochleagram_1_10_secs', (211, 1950)) # 10 seconds: 211 freq, 1950 time bins
    }
    
    # Get duration from kwargs, default to 2 seconds
    duration = ds_kwargs.get('duration', 2)
    
    # Get the appropriate cochleagram configuration and dimensions
    if duration not in duration_to_config:
        raise ValueError(f"Unsupported duration: {duration}. Supported durations are {', '.join(map(str, duration_to_config.keys()))} seconds.")
    
    audio_representation, coch_size = duration_to_config[duration]
    
    ds = jsinV3(JSIN_PATH, include_rep_in_model=include_rep_in_model, 
                audio_representation=audio_representation,
                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                include_identity_sequential=include_identity_sequential, 
                eval_max=8,
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = None

    # Spectemp Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
        #  'filtered_signal', 
        #  'spectempfilter_power',
         'avgpool',
        #  'final',
    ]

    # Restore the model
    model, _ = make_and_restore_model(arch='spectemp_filts_time_average_coch1', 
                                      dataset=ds, 
                                      parallel=False,
                                      resume_path=resume_path,
                                      strict=strict,
                                      arch_kwargs={'coch_size': coch_size}
                                     )

    # send the model to the GPU and return it. 
    model.cuda()
    model.eval()

    model.model.filter_freqs = model.model._modules['1'].spectempfilterbank.spec_temp_freqs
    model.model.coarse_define_layers = ['filtered_signal', 'spectempfilter_power']

    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(include_rep_in_model=True,
         use_normalization_for_audio_rep=False,
         return_metamer_layers=False,
         include_identity_sequential=False,
         strict=True,
         ds_kwargs={}):
    # This parameter is not used for this model
    print(f"ds_kwargs inside build_network.main: {ds_kwargs}")

    if return_metamer_layers:
        model, ds, metamer_layers = build_net(include_rep_in_model=include_rep_in_model,
                                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                                              return_metamer_layers=return_metamer_layers,
                                              strict=strict,
                                              include_identity_sequential=include_identity_sequential,
                                              ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(include_rep_in_model=include_rep_in_model,
                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                              return_metamer_layers=return_metamer_layers,
                              strict=strict,
                              include_identity_sequential=include_identity_sequential,
                              ds_kwargs=ds_kwargs)
        return model, ds

if __name__== "__main__":
    main()