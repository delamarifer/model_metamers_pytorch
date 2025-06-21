import torch
from math import floor
import random

class InversionLossLayer(torch.nn.Module):
    """Loss used for most metamer generation experiments"""
    def __init__(self, layer_to_invert, fake_relu=True, normalize_loss=False):
        super(InversionLossLayer, self).__init__()
        self.layer_to_invert = layer_to_invert
        self.fake_relu = fake_relu
        self.normalize_loss = normalize_loss

    def forward(self, model, inp, targ):
        _, _, all_outputs = model(inp, with_latent=True, fake_relu=self.fake_relu)
        rep = all_outputs[self.layer_to_invert].contiguous().view(all_outputs[self.layer_to_invert].size(0), -1)
        if self.normalize_loss:
            loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
        else:
            loss = torch.norm(rep - targ, dim=1)
        return loss, None


class InversionLossLayerWithRandomSingleUnitOptimizationDropout(torch.nn.Module):
    """Only include a single unit of the layer in each optimization pass"""
    def __init__(self, layer_to_invert, fake_relu=True, normalize_loss=False,
                 optimization_step=50):
        super(InversionLossLayerWithRandomSingleUnitOptimizationDropout, self).__init__()
        self.layer_to_invert = layer_to_invert
        self.fake_relu = fake_relu
        self.normalize_loss = normalize_loss
        self.enable_dropout_flag = False
        self.optimization_count = 0
        self.optimization_step = optimization_step
        self.selected_unit = 0 # Start with unit 0 as a placeholder. 
        self.all_units_count = 400

    def _enable_dropout_functions(self):
        self.enable_dropout_flag = True

    def _disable_dropout_functions(self):
        self.enable_dropout_flag = False

    def forward(self, model, inp, targ):
        _, _, all_outputs = model(inp, with_latent=True, fake_relu=self.fake_relu)

        rep = all_outputs[self.layer_to_invert]
        if isinstance(targ, list):
            assert isinstance(rep, list), 'targ is list but rep is not. this is not supported.'
        else:
            targ = targ.view(rep.shape)

        # Choose which unit to optimize
        if (self.optimization_count%self.optimization_step == 0) and (self.optimization_count<self.all_units_count):
            if isinstance(targ, list):
                self.selected_unit = random.randint(0, len(targ)-1)
                print('Selected Unit is %d'%self.selected_unit)
            elif len(targ.shape)==4:
                self.selected_unit = random.randint(0,targ.shape[1]-1)
                print('Selected Unit is %d'%self.selected_unit)
            elif len(targ.shape)==3: # Also try dropout for the c2 layer
                self.selected_unit = random.randint(0,targ.shape[1]-1)
                print('Selected Unit is %d'%self.selected_unit)

        if self.enable_dropout_flag:
            self._enable_dropout_functions()
            self.optimization_count+=1
        else:
            self._disable_dropout_functions()

        # In this case we know that we have units to drop out
        if isinstance(targ, list):
            if self.enable_dropout_flag and (self.optimization_count<self.all_units_count):
                targ = targ[self.selected_unit]
                rep = rep[self.selected_unit]
            else:
                targ = torch.cat([a.view(a.size(0), -1) for a in targ],1)
                print(targ.shape)
                rep = torch.cat([b.view(b.size(0), -1) for b in rep], 1)
                print(rep.shape)
        elif (len(targ.shape) == 4) and self.enable_dropout_flag and (self.optimization_count<self.all_units_count):
            targ = targ[:,self.selected_unit,:,:]
            rep = rep[:,self.selected_unit,:,:]
        elif (len(targ.shape) == 3) and self.enable_dropout_flag and (self.optimization_count<self.all_units_count):
            targ = targ[:,self.selected_unit,:]
            rep = rep[:,self.selected_unit,:]

        targ = targ.contiguous().view(targ.size(0), -1)
        rep = rep.contiguous().view(rep.size(0), -1)

        if self.normalize_loss:
            loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
        else:
            loss = torch.norm(rep - targ, dim=1)
        if self.optimization_count%100 == 0:
            print(loss)
        return loss, None


class InversionLossLayerWithCoarseDefineSpecTemp111(torch.nn.Module):
    """Only include a subset of the units of the layer in each optimization pass,
    with a particular order. This loss is only set up for the SpecTemp model"""
    def __init__(self, layer_to_invert, fake_relu=True, normalize_loss=False,
                 optimization_step=400):
        super(InversionLossLayerWithCoarseDefineSpecTemp111, self).__init__()
        self.layer_to_invert = layer_to_invert
        self.fake_relu = fake_relu
        self.normalize_loss = normalize_loss
        self.optimization_count = 0
        self.optimization_step = optimization_step
        # Use these as the cutoffs. After optimization_step iterations choose
        # different filters to include in the optimization
        self.filter_cutoffs = [[0, 0],
                  [0.5, 0.0625],
                  [1, 0.125],
                  [2, 0.25],
                  [4, 0.5],
                  [8, 1],
                  [16, 2],
                 ]
        self.cutoff_idx = floor(self.optimization_count / self.optimization_step)

        self.enable_dropout_flag = False

    def _enable_dropout_functions(self):
        self.enable_dropout_flag = True

    def _disable_dropout_functions(self):
        self.enable_dropout_flag = False

    def forward(self, model, inp, targ):
        if self.enable_dropout_flag:
            self.optimization_count+=1
            self.cutoff_idx = min(floor(self.optimization_count / self.optimization_step), len(self.filter_cutoffs)-1)
            self._enable_dropout_functions()
        else:
            self._disable_dropout_functions()

        _, _, all_outputs = model(inp, with_latent=True, fake_relu=self.fake_relu)
        rep = all_outputs[self.layer_to_invert]
        targ = targ.view(rep.shape)

        if self.layer_to_invert in model.coarse_define_layers:
            # Select a subset of the units for some of the layers
            all_freqs = model.filter_freqs
            include_filts = (abs(all_freqs[:,0]) <= self.filter_cutoffs[self.cutoff_idx][0]) + (abs(all_freqs[:,1]) <= self.filter_cutoffs[self.cutoff_idx][1])

            targ = targ[:,include_filts,:,:]
            rep = rep[:,include_filts,:,:]

        targ = targ.contiguous().view(targ.size(0), -1)
        rep = rep.contiguous().view(rep.size(0), -1)

        if self.normalize_loss:
            loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
            if self.optimization_count%100==0:
                print(loss)
        else:
            loss = torch.norm(rep - targ, dim=1)
        return loss, None


class TimeAveragedInversionLossLayer(torch.nn.Module):
    """
    Enhanced loss layer that handles both single-layer and multi-layer (cumulative) losses
    with time averaging and squared activation support.

    FUNCTIONALITY:
    1. If 'targ' is a dict, computes + sums the per-layer loss for each key
    2. If 'targ' is a single tensor, uses self.layer_to_invert for single-layer loss
    3. Optionally time-averages the squared representation along specified dimension
    4. Supports skipping time averaging for specific layers
    5. Normalizes loss if specified
    """
    def __init__(self, layer_to_invert=None, fake_relu=False, normalize_loss=True,
                 dim_to_average=-1, skip_time_average_layers=None, debug=True):
        super(TimeAveragedInversionLossLayer, self).__init__()
        self.layer_to_invert = layer_to_invert
        self.fake_relu = fake_relu
        self.normalize_loss = normalize_loss
        self.dim_to_average = dim_to_average
        self.debug = debug
        
        # Default layers to skip time averaging
        self.skip_time_average_layers = skip_time_average_layers or [
            'input_after_preproc_cochleagrams',
            'avgpool',
            'final/signal/word_int',
            'final/signal/speaker_int',
            'final/noise/labels_binary_via_int'
        ]

    def _process_representations(self, rep_raw, targ_raw, layer_name):
        """Helper function to process representations with time averaging"""
        if self.debug:
            print(f"\nProcessing layer: {layer_name}")
            print(f"Raw representation shape: {rep_raw.shape}")
            print(f"Raw target shape: {targ_raw.shape}")
            print(f"Time dimension (dim {self.dim_to_average}) size: {rep_raw.shape[self.dim_to_average]}")
            
        if layer_name in self.skip_time_average_layers:
            if self.debug:
                print(f"Skipping time average for layer {layer_name}")
            # Skip time averaging, just flatten
            rep_processed = rep_raw.view(rep_raw.size(0), -1)
            targ_processed = targ_raw.view(targ_raw.size(0), -1)
        else:
            if self.debug:
                print(f"Applying time average and squared activation")
                print(f"Before squaring - rep mean: {rep_raw.mean().item():.4f}, std: {rep_raw.std().item():.4f}")
            
            # Time average squared activations
            rep_squared = rep_raw**2
            targ_squared = targ_raw**2
            
            if self.debug:
                print(f"After squaring - rep mean: {rep_squared.mean().item():.4f}, std: {rep_squared.std().item():.4f}")
            
            rep_processed = torch.mean(rep_squared, dim=self.dim_to_average)
            targ_processed = torch.mean(targ_squared, dim=self.dim_to_average)
            
            if self.debug:
                print(f"After time averaging - rep mean: {rep_processed.mean().item():.4f}, std: {rep_processed.std().item():.4f}")
                print(f"Processed shapes - rep: {rep_processed.shape}, targ: {targ_processed.shape}")
            
            rep_processed = rep_processed.view(rep_processed.size(0), -1)
            targ_processed = targ_processed.view(targ_processed.size(0), -1)
            
            if self.debug:
                print(f"Final flattened shapes - rep: {rep_processed.shape}, targ: {targ_processed.shape}")
        
        return rep_processed, targ_processed

    def _compute_loss(self, rep_processed, targ_processed):
        """Helper function to compute normalized or unnormalized loss"""
        if self.normalize_loss:
            # Add small epsilon to avoid division by zero
            norm_diff = torch.norm(rep_processed - targ_processed, dim=1)
            norm_targ = torch.norm(targ_processed, dim=1) + 1e-8
            loss = torch.div(norm_diff, norm_targ)
            if self.debug:
                print(f"\nLoss computation:")
                print(f"Norm of difference: {norm_diff.mean().item():.4f}")
                print(f"Norm of target: {norm_targ.mean().item():.4f}")
                print(f"Final normalized loss: {loss.mean().item():.4f}")
        else:
            loss = torch.norm(rep_processed - targ_processed, dim=1)
            if self.debug:
                print(f"\nLoss computation:")
                print(f"Unnormalized loss: {loss.mean().item():.4f}")
        return loss

    def forward(self, model, inp, targ):
        # Get all intermediate representations (including any extra outputs)
        _, _, all_outputs, *_ = model(inp, with_latent=True, fake_relu=self.fake_relu)

        # Handle dictionary of targets (cumulative loss)
        if isinstance(targ, dict):
            total_loss = torch.zeros(1, device=inp.device)
            for layer_name, targ_output in targ.items():
                rep_raw = all_outputs[layer_name]
                targ_raw = targ_output.view_as(rep_raw)
                
                # Process representations
                rep_processed, targ_processed = self._process_representations(
                    rep_raw, targ_raw, layer_name)
                
                # Compute and accumulate loss
                layer_loss = self._compute_loss(rep_processed, targ_processed)
                total_loss += layer_loss.mean()
            
            return total_loss, None

        # Handle single target tensor
        else:
            if self.layer_to_invert is None:
                raise ValueError("layer_to_invert must be specified for single-layer loss")
            
            rep_raw = all_outputs[self.layer_to_invert]
            targ_raw = targ.view_as(rep_raw)
            
            # Process representations
            rep_processed, targ_processed = self._process_representations(
                rep_raw, targ_raw, self.layer_to_invert)
            
            # Compute loss
            loss_vals = self._compute_loss(rep_processed, targ_processed)
            return loss_vals, None


LOSSES = {
    'inversion_loss_layer': InversionLossLayer,
    'coarse_define_spectemp_inversion_loss_layer':InversionLossLayerWithCoarseDefineSpecTemp111,
    'random_single_unit_optimization_inversion_loss_layer': InversionLossLayerWithRandomSingleUnitOptimizationDropout,
    'time_averaged_inversion_loss_layer': TimeAveragedInversionLossLayer,
}
'''
Dictionary of loss functions for synthesis. A loss function class can be accessed as

>>> import robustness.custom_synthesis_losses
>>> ds = custom_synthesis_losses.LOSSES['inversion_loss_layer'](<LOSS ARGS>)
'''
