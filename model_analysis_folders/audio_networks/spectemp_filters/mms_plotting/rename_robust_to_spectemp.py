import os

root = 'model_analysis_folders/audio_networks/spectemp_filters/metamers_by_run/metamers_1/natural_sounds_norman_haignere_coarse_define_spectemp_inversion_loss_layer_RS0_I3000_N8_LR1.000_DECAY0.500_ROBUST'

for dirpath, dirnames, filenames in os.walk(root, topdown=False):
    # Rename files
    for fname in filenames:
        if 'ROBUST' in fname:
            old = os.path.join(dirpath, fname)
            new = os.path.join(dirpath, fname.replace('ROBUST', 'SPECTEMP'))
            os.rename(old, new)
    # Rename directories
    for dname in dirnames:
        if 'ROBUST' in dname:
            old = os.path.join(dirpath, dname)
            new = os.path.join(dirpath, dname.replace('ROBUST', 'SPECTEMP'))
            os.rename(old, new) 