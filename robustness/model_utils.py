import os
import dill
import torch as ch

# External / local imports ----------------------------------------------------
from .attacker import AttackerModel            # robust audio Attacker wrapper
from .tools import helpers, constants          # helpers provides Parameters, ckpt_at_epoch
import cox                                     # experiment I/O

# -----------------------------------------------------------------------------
#                           SMALL HELPER CLASSES
# -----------------------------------------------------------------------------
class FeatureExtractor(ch.nn.Module):
    """Wraps a *submod* and exposes intermediate activations as output."""
    def __init__(self, submod, layers):
        super().__init__()
        self.submod = submod
        self.layers = layers
        for fn in layers:
            layer = fn(self.submod)
            layer.register_forward_hook(lambda m, _i, o: m.register_buffer("activations", o))
    def forward(self, *args, **kwargs):
        out = self.submod(*args, **kwargs)
        activs = [fn(self.submod).activations for fn in self.layers]
        return [out] + activs

# -----------------------------------------------------------------------------
#                            CHECKPOINT LOADING
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
#                        PUBLIC FACTORY / LOADER
# -----------------------------------------------------------------------------

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False, strict=True,
         remap_checkpoint_keys={}, append_name_front_keys=None,
         change_prefix_checkpoint={}, arch_kwargs={}, model_type="standard"):
    """Construct AttackerModel and (optionally) load a checkpoint."""
    base_model = dataset.get_model(arch, pytorch_pretrained, arch_kwargs) if isinstance(arch, str) else arch
    model      = AttackerModel(base_model, dataset)
    checkpoint = None
    if resume_path:
        if not os.path.isfile(resume_path):
            raise ValueError(f"=> No checkpoint found at '{resume_path}'")
        print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = ch.load(resume_path, pickle_module=dill, map_location="cpu")
        state_key = "model" if "model" in checkpoint else "state_dict"
        sd = checkpoint.get(state_key, checkpoint)
        # remove DataParallel prefix
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        # prepend optional prefixes
        if append_name_front_keys:
            aug = {}
            for p in append_name_front_keys:
                aug.update({p + k: v for k, v in sd.items()})
            sd = aug
        sd = _remap_state_dict(sd)
        # manual remaps
        for old, new in remap_checkpoint_keys.items():
            if isinstance(new, list):
                for nk in new: sd[nk] = sd[old]
                del sd[old]
            else:
                sd[new] = sd.pop(old)
        for old_pref, new_pref in change_prefix_checkpoint.items():
            for k in list(sd.keys()):
                if isinstance(old_pref, int) or k.startswith(old_pref):
                    nk = (new_pref + k) if isinstance(old_pref, int) else new_pref + k[len(old_pref):]
                    sd[nk] = sd.pop(k)
        print("=> first 10 remapped keys:", list(sd.keys())[:10])
        try:
            model.load_state_dict(sd, strict=strict)
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e):
                missing = {ln.strip() for ln in str(e).split("\n") if ln.strip().startswith("\"")}
                if missing.issubset(MISSING_AUDIO_BUFFERS):
                    print("=> only cochleagram buffers missing – retry with strict=False")
                    model.load_state_dict(sd, strict=False)
                else:
                    raise
            else:
                raise
        if parallel:
            model = ch.nn.DataParallel(model)
        model = model.cuda()
        print(f"=> checkpoint loaded (epoch {checkpoint.get('epoch','?')})")
    return model, checkpoint

# -----------------------------------------------------------------------------
#               Convenience: recover model + dataset from a cox Store
# -----------------------------------------------------------------------------

def model_dataset_from_store(store, overwrite_params=None, which="last"):
    """Return (model, dataset, args) from a *cox* Store or (root, exp_name)."""
    overwrite_params = overwrite_params or {}
    if isinstance(store, tuple):
        root, exp = store
        store = cox.store.Store(root, exp, mode="r")
    meta = store["metadata"]
    df   = meta.df.to_dict()
    args = {k: v[0] for k, v in df.items()}
    for k in list(args.keys()):
        if meta.schema[k] == store.OBJECT:
            args[k] = meta.get_object(k)
        elif meta.schema[k] == store.PICKLE:
            args[k] = meta.get_pickle(k)
    args.update(overwrite_params)
    args = helpers.Parameters(args)
    data_path = os.path.expandvars(args.data) or "/tmp/"
    dataset   = DATASETS[args.dataset](data_path)
    if which == "last":
        ckpt_path = os.path.join(store.path, constants.CKPT_NAME)
    elif which == "best":
        ckpt_path = os.path.join(store.path, constants.CKPT_NAME_BEST)
    else:
        ckpt_path = os.path.join(store.path, helpers.ckpt_at_epoch(which))
    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=ckpt_path, parallel=False)
    return model, dataset, args
