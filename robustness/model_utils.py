import torch as ch
import dill
import os
from .tools import helpers, constants
from .attacker import AttackerModel

# Audio-specific constants for handling missing buffers
MISSING_AUDIO_BUFFERS = {
    "preproc.audio_preproc.full_rep.rep.downsampling_op.downsample_filter",
    "preproc.audio_preproc.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
    "preproc.audio_preproc.full_rep.rep.Cochleagram.downsampling.downsample_filter",
    "attacker.preproc.audio_preproc.full_rep.rep.downsampling_op.downsample_filter",
    "attacker.preproc.audio_preproc.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
    "attacker.preproc.audio_preproc.full_rep.rep.Cochleagram.downsampling.downsample_filter",
}

def _remap_state_dict(sd):
    out = {}
    for k, v in sd.items():
        # Skip cochleagram filter parameters
        if k in [
            "preproc.audio_preproc.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
            "attacker.preproc.audio_preproc.full_rep.rep.Cochleagram.compute_subbands.coch_filters"
        ]:
            continue
            
        if k.startswith("model.0."):
            nk = k.replace("model.0.", "preproc.audio_preproc.")
        elif k.startswith("model.1."):
            nk = k.replace("model.1.", "model.")
        elif k.startswith("attacker.model.0."):
            nk = k.replace("attacker.model.0.", "attacker.preproc.audio_preproc.")
        elif k.startswith("attacker.model.1."):
            nk = k.replace("attacker.model.1.", "attacker.model.")
        elif k.startswith("model.full_rep."):
            nk = k.replace("model.", "preproc.audio_preproc.", 1)
        elif k.startswith("attacker.model.full_rep."):
            nk = k.replace("attacker.model.", "attacker.preproc.audio_preproc.", 1)
        else:
            nk = k
        out[nk] = v
    return out

class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False, strict=True,
         remap_checkpoint_keys={}, append_name_front_keys=None,
         change_prefix_checkpoint={},
         arch_kwargs={}, model_type="standard"):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint
        parallel (bool): if True, wrap the model in a DataParallel 
            (default True, recommended)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        strict (bool): If true, the state dict must exactly match, if False
            loading ignores non-matching keys
        remap_checkpoint_keys (dict): Modifies keys in the loaded state_dict 
            to new names, so that we can load old models if the code has changed. 
        append_name_front_keys (list): if not none, for each element of the list 
            makes new keys in the state dict that have the element appended to the front
            of the name. Useful for transfer models, if they were saved without the attacker model class. 
        change_prefix_checkpoint (dict) : for each element {old_prefix:new_prefix},
            checks the state dict for entries that start with old_prefix and changes
            that portion of the key to new_prefix.
        model_type (str): Type of model being loaded (default: "standard")
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    classifier_model = dataset.get_model(arch, pytorch_pretrained, arch_kwargs) if \
                            isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path, pickle_module=dill)
            
            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            try:
                sd = checkpoint[state_dict_path]
            except:
                print('Missing state dict key %s from checkpoint. Assuming checkpoint is simply the state dictionary and we do not need a key'%state_dict_path)
                sd = checkpoint

            if append_name_front_keys is not None:
                new_sd = {}
                for key_idx in range(len(append_name_front_keys)):
                    sd_temp = {'%s%s'%(append_name_front_keys[key_idx], k):v for k,v in sd.items()}
                    new_sd.update(sd_temp)
                sd = new_sd

            sd = {k[len('module.'):]:v for k,v in sd.items()}

            # Filter out cochleagram parameters before any renaming
            sd = {k: v for k, v in sd.items() if not any(
                x in k for x in [
                    "full_rep.rep.Cochleagram.compute_subbands.coch_filters",
                    "full_rep.rep.downsampling_op.downsample_filter",
                    "full_rep.rep.Cochleagram.downsampling.downsample_filter",
                    "model.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
                    "model.full_rep.rep.downsampling_op.downsample_filter",
                    "model.full_rep.rep.Cochleagram.downsampling.downsample_filter",
                    "model.0.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
                    "model.0.full_rep.rep.downsampling_op.downsample_filter",
                    "model.0.full_rep.rep.Cochleagram.downsampling.downsample_filter",
                    "model.1.full_rep.rep.Cochleagram.compute_subbands.coch_filters",
                    "model.1.full_rep.rep.downsampling_op.downsample_filter",
                    "model.1.full_rep.rep.Cochleagram.downsampling.downsample_filter"
                ]
            )}

            # The following blocks are used in specific cases where we are loading models that are trained with different
            # module names than are used in this library.

            # Load models if the keys changed slightly
            for old_key, new_key in remap_checkpoint_keys.items():
                print('mapping %s to %s'%(old_key, new_key))
                if type(new_key) is list: # If there are multiple keys that should be the same value (ie with attacker model)
                    for new_key_temp in new_key:
                        sd[new_key_temp] = sd[old_key]
                    del sd[old_key]
                else:
                    sd[new_key]=sd.pop(old_key)
            # Swaps out a prefix
            for old_prefix, new_prefix in change_prefix_checkpoint.items():
                sd_keys_temp = list(sd.keys())
                for sd_key in sd_keys_temp:
                    if type(old_prefix)==int: # If we need to add prefix to EVERYTHING
                        sd[new_prefix + sd_key] = sd[sd_key]
                        del sd[sd_key]
                    else:
                        if sd_key.startswith(old_prefix):
                            sd[new_prefix + sd_key.split(old_prefix)[-1]] = sd[sd_key]
                            del sd[sd_key]

            # Apply audio-specific state dict remapping
            sd = _remap_state_dict(sd)

            try:
                model.load_state_dict(sd, strict=strict)
            except RuntimeError as e:
                if "Missing key(s) in state_dict" in str(e):
                    missing = {ln.strip() for ln in str(e).split("\n") if ln.strip().startswith("\"")}
                    if missing.issubset(MISSING_AUDIO_BUFFERS):
                        print("=> only cochleagram buffers missing – retry with strict=True")
                        model.load_state_dict(sd, strict=True)
                    else:
                        raise
                else:
                    raise

            if parallel:
                model = ch.nn.DataParallel(model)
            model = model.cuda()
 
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint.get('epoch', 'epoch number not found')))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint

def model_dataset_from_store(s, overwrite_params={}, which='last'):
    '''
    Given a store directory corresponding to a trained model, return the
    original model, dataset object, and args corresponding to the arguments.
    '''
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data) or '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args