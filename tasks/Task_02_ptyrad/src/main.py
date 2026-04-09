import os
import sys
from copy import deepcopy
from math import prod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.fft import fft2, ifft2
from torchvision.transforms.functional import gaussian_blur

# =============================================================================
# UTILITY FUNCTIONS AND IMPORTS FROM PTYRAD
# =============================================================================

from ptyrad.load import load_params
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger, vprint, time_sync, imshift_batch, torch_phasor
from ptyrad.constraints import CombinedConstraint
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.forward import multislice_forward_model_vec_all
from ptyrad.reconstruction import create_optimizer, prepare_recon, parse_sec_to_time_str, recon_loop, IndicesDataset


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(params_path, gpuid=0, logger=None):
    """
    Loads data from disk/arguments and returns preprocessed tensors/arrays.
    
    Args:
        params_path: Path to the YAML parameters file
        gpuid: GPU device ID (None for CPU)
        logger: Optional logger instance
        
    Returns:
        dict containing:
            - params: Loaded and validated parameters
            - device: PyTorch device
            - logger: Logger instance
            - init: Initialized variables from Initializer
            - loss_fn: Combined loss function
            - constraint_fn: Combined constraint function
    """
    print_system_info()
    
    # Load and validate parameters
    params = load_params(params_path, validate=True)
    device = set_gpu_device(gpuid=gpuid)
    
    verbose = not params['recon_params']['if_quiet']
    
    # Initialize the Initializer
    vprint("### Initializing Initializer ###")
    init = Initializer(params['init_params'], seed=None).init_all()
    vprint(" ")
    
    # Initialize loss function
    vprint("### Initializing loss function ###")
    loss_params = params['loss_params']
    vprint("Active loss types:")
    for key, value in loss_params.items():
        if value.get('state', False):
            vprint(f"  {key.ljust(12)}: {value}")
    loss_fn = CombinedLoss(loss_params, device=device)
    vprint(" ")
    
    # Initialize constraint function
    vprint("### Initializing constraint function ###")
    constraint_params = params['constraint_params']
    vprint("Active constraint types:")
    for key, value in constraint_params.items():
        if value.get('start_iter', None) is not None:
            vprint(f"  {key.ljust(14)}: {value}")
    constraint_fn = CombinedConstraint(constraint_params, device=device, verbose=verbose)
    vprint(" ")
    
    vprint("### Done initializing data and preprocessing ###")
    vprint(" ")
    
    return {
        'params': params,
        'device': device,
        'logger': logger,
        'init': init,
        'loss_fn': loss_fn,
        'constraint_fn': constraint_fn,
        'verbose': verbose
    }


# =============================================================================
# 2. FORWARD OPERATOR (PHYSICS MODEL)
# =============================================================================

class PtychoAD(torch.nn.Module):
    """
    Main optimization class for ptychographic reconstruction using automatic differentiation (AD).
    
    This class implements the forward operator A in y = A(x), representing the physical model
    of ptychographic imaging.
    """

    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            
            vprint('### Initializing PtychoAD model ###', verbose=verbose)
            
            # Setup model behaviors
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params['detector_blur_std']
            self.obj_preblur_std = model_params['obj_preblur_std']
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Parse the learning rate and start iter for optimizable tensors
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params.get('start_iter')
                end_iter_dict[key] = params.get('end_iter')
                lr_dict[key] = params['lr']
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = start_iter_dict
            self.end_iter = end_iter_dict
            self.lr_params = lr_dict
            
            # Optimizable parameters
            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))
            
            # Buffers are used during forward pass
            self.register_buffer('omode_occu', torch.tensor(init_variables['omode_occu'], dtype=torch.float32, device=device))
            self.register_buffer('H', torch.tensor(init_variables['H'], dtype=torch.complex64, device=device))
            self.register_buffer('measurements', torch.tensor(init_variables['measurements'], dtype=torch.float32, device=device))
            self.register_buffer('N_scan_slow', torch.tensor(init_variables['N_scan_slow'], dtype=torch.int32, device=device))
            self.register_buffer('N_scan_fast', torch.tensor(init_variables['N_scan_fast'], dtype=torch.int32, device=device))
            self.register_buffer('crop_pos', torch.tensor(init_variables['crop_pos'], dtype=torch.int32, device=device))
            self.register_buffer('slice_thickness', torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.register_buffer('dx', torch.tensor(init_variables['dx'], dtype=torch.float32, device=device))
            self.register_buffer('dk', torch.tensor(init_variables['dk'], dtype=torch.float32, device=device))
            self.register_buffer('lambd', torch.tensor(init_variables['lambd'], dtype=torch.float32, device=device))
            
            self.random_seed = init_variables['random_seed']
            self.length_unit = init_variables['length_unit']
            self.scan_affine = init_variables['scan_affine']
            self.tilt_obj = bool(self.lr_params['obj_tilts'] != 0 or torch.any(self.opt_obj_tilts))
            self.shift_probes = bool(self.lr_params['probe_pos_shifts'] != 0)
            self.change_thickness = bool(self.lr_params['slice_thickness'] != 0)
            self.probe_int_sum = self.get_complex_probe_view().abs().pow(2).sum()
            self.loss_iters = []
            self.iter_times = []
            self.dz_iters = []
            self.avg_tilt_iters = []

            # Create grids for shifting
            self.create_grids()

            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obja': self.opt_obja,
                'objp': self.opt_objp,
                'obj_tilts': self.opt_obj_tilts,
                'slice_thickness': self.opt_slice_thickness,
                'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self.create_optimizable_params_dict(self.lr_params, self.verbose)

            # Initialize propagator-related variables
            self.init_propagator_vars()
            
            # Initialize iteration numbers that require torch.compile
            self.init_compilation_iters()
            
            vprint('### Done initializing PtychoAD model ###', verbose=verbose)
            vprint(' ', verbose=verbose)
            
    def get_complex_probe_view(self):
        """Retrieve complex view of the probe"""
        return torch.view_as_complex(self.opt_probe)
        
    def create_grids(self):
        """Create the grids for shifting probes, selecting obj ROI, and Fresnel propagator"""
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:]
        Noy, Nox = self.opt_objp.shape[-2:]
        
        # Grids for Fresnel propagator
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky, Kx], dim=0)
        
        # Grids for obj_ROI selection
        rpy, rpx = torch.meshgrid(
            torch.arange(Npy, dtype=torch.int32, device=device),
            torch.arange(Npx, dtype=torch.int32, device=device),
            indexing='ij'
        )
        self.rpy_grid = rpy
        self.rpx_grid = rpx
        
        # Grids for shifting probes and objects
        kpy, kpx = torch.meshgrid(
            torch.fft.fftfreq(Npy, dtype=torch.float32, device=device),
            torch.fft.fftfreq(Npx, dtype=torch.float32, device=device),
            indexing='ij'
        )
        koy, kox = torch.meshgrid(
            torch.fft.fftfreq(Noy, dtype=torch.float32, device=device),
            torch.fft.fftfreq(Nox, dtype=torch.float32, device=device),
            indexing='ij'
        )
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0)
        self.shift_object_grid = torch.stack([koy, kox], dim=0)
    
    def create_optimizable_params_dict(self, lr_params, verbose=True):
        """Sets the optimizer with lr_params"""
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"WARNING: '{param_name}' is not a valid parameter name")
            else:
                self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] == 1)
                if lr != 0:
                    self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
        if verbose:
            self.print_model_summary()
        
    def init_propagator_vars(self):
        """Initialize propagator related variables"""
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid
        tilts_y_full = self.opt_obj_tilts[:, 0, None, None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:, 1, None, None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full)))
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2)
    
    def init_compilation_iters(self):
        """Initialize iteration numbers that require torch.compile"""
        compilation_iters = {1}
        for param_name in self.optimizable_tensors.keys():
            start_iter = self.start_iter.get(param_name)
            end_iter = self.end_iter.get(param_name)
            if start_iter is not None and start_iter >= 1:
                compilation_iters.add(start_iter)
            if end_iter is not None and end_iter >= 1:
                compilation_iters.add(end_iter)
        self.compilation_iters = sorted(compilation_iters)
        
    def print_model_summary(self):
        """Prints a summary of the model's optimizable variables and statistics."""
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}")
        total_var = sum(tensor.numel() for _, tensor in self.optimizable_tensors.items() if tensor.requires_grad)
        vprint(" ")
        
        vprint('### Optimizable variables statitsics ###')
        vprint(f"Total measurement values  : {self.measurements.numel():,d}")
        vprint(f"Total optimizing variables: {total_var:,d}")
        vprint(f"Overdetermined ratio      : {self.measurements.numel()/total_var:.2f}")
        vprint(" ")
        
        vprint('### Model behavior ###')
        vprint(f"Obj preblur               : {True if self.obj_preblur_std is not None else False}")
        vprint(f"Tilt propagator           : {self.tilt_obj}")
        vprint(f"Change slice thickness    : {self.change_thickness}")
        vprint(f"Sub-px probe shift        : {self.shift_probes}")
        vprint(f"Detector blur             : {True if self.detector_blur_std is not None else False}")
        vprint(f"On-the-fly meas padding   : {True if self.meas_padded is not None else False}")
        vprint(f"On-the-fly meas resample  : {True if self.meas_scale_factors is not None else False}")
        vprint(" ")
    
    def get_obj_ROI(self, indices):
        """Get object ROI with integer coordinates"""
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None, :, :] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None, :, :] + self.crop_pos[indices, None, None, 1]
        object_roi = opt_obj[:, :, obj_ROI_grid_y, obj_ROI_grid_x, :].permute(2, 0, 1, 3, 4, 5)
        return object_roi
    
    def get_obj_patches(self, indices):
        """Get object patches from specified indices"""
        object_patches = self.get_obj_ROI(indices)
        
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_patches
        else:
            obj = object_patches.permute(5, 0, 1, 2, 3, 4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            object_patches = gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1, 2, 3, 4, 5, 0)
            return object_patches
        
    def get_probes(self, indices):
        """Get probes for each position"""
        probe = self.get_complex_probe_view()
        
        if self.shift_probes:
            probes = imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        else:
            probes = torch.broadcast_to(probe, (indices.shape[0], *probe.shape))
        return probes
    
    def get_propagators(self, indices):
        """Get propagators for each position"""
        tilt_obj = self.tilt_obj
        global_tilt = (self.opt_obj_tilts.shape[0] == 1)
        change_tilt = (self.lr_params['obj_tilts'] != 0)
        change_thickness = self.change_thickness
        
        dz = self.opt_slice_thickness
        Kz = self.Kz
        Ky, Kx = self.propagator_grid
        
        if global_tilt:
            tilts = self.opt_obj_tilts
        else:
            tilts = self.opt_obj_tilts[indices]
        tilts_y = tilts[:, 0, None, None] / 1e3
        tilts_x = tilts[:, 1, None, None] / 1e3
                
        if tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif tilt_obj and not change_thickness:
            if change_tilt:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz[None,]
        else:
            return self.H[None,]

    def get_propagated_probe(self, index):
        """Get propagated probe through slices"""
        probe = self.get_probes(index)[0].detach()
        H = self.get_propagators(index)[[0]].detach()
        n_slices = self.opt_objp.shape[1]
        probe_prop = torch.zeros((n_slices, *probe.shape), dtype=probe.dtype, device=probe.device)
        
        psi = probe
        for n in range(n_slices):
            probe_prop[n] = psi
            psi = ifft2(H[None,] * fft2(psi))
        
        return probe_prop
    
    def get_forward_meas(self, object_patches, probes, propagators):
        """Get forward model measurements"""
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
            
        return dp_fwd
    
    def get_measurements(self, indices=None):
        """Get measurements for each position"""
        measurements = self.measurements
        device = self.device
        dtype = measurements.dtype
        if self.meas_padded is not None:
            meas_padded = self.meas_padded
            meas_padded_idx = self.meas_padded_idx
            pad_h1, pad_h2, pad_w1, pad_w2 = meas_padded_idx
        scale_factor = tuple(self.meas_scale_factors) if self.meas_scale_factors is not None else None
        
        if indices is not None:
            measurements = self.measurements[indices]
            
            if self.meas_padded is not None:
                canvas = torch.zeros((measurements.shape[0], *meas_padded.shape[-2:]), dtype=dtype, device=device)
                canvas += meas_padded
                canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
                measurements = canvas
            
            if self.meas_scale_factors is not None and any(factor != 1 for factor in scale_factor):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale_factor, mode='bilinear')[0]
                measurements = measurements / prod(scale_factor)
        else:
            if self.meas_padded is not None or self.meas_scale_factors is not None:
                vprint(f"WARNING: 'on-the-fly' measurements padding/resampling detected, but they are ignored.")
            measurements = self.measurements
        
        return measurements
    
    def clear_cache(self):
        """Clear temporary attributes like cached object patches."""
        self._current_object_patches = None
        
    def forward(self, indices):
        """
        Forward operator: Computes predicted diffraction patterns from the current model state.
        
        This is the core physics model A in y = A(x), where:
        - x: latent variables (object amplitude/phase, probe, etc.)
        - y: predicted diffraction patterns
        
        Args:
            indices: Batch indices for which to compute forward model
            
        Returns:
            dp_fwd: Predicted diffraction patterns (N, Ky, Kx)
        """
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        dp_fwd = self.get_forward_meas(object_patches, probes, propagators)
        
        # Keep the object_patches for later object-specific loss
        self._current_object_patches = object_patches
        
        return dp_fwd


def forward_operator(model, indices):
    """
    Wrapper for the forward operator that represents the physical model A in y = A(x).
    
    This function computes predicted observations from the latent variables (object, probe, etc.)
    using the multislice ptychographic forward model.
    
    Args:
        model: PtychoAD model containing the latent variables
        indices: Batch indices for which to compute forward model
        
    Returns:
        y_pred: Predicted diffraction patterns (N, Ky, Kx)
    """
    return model(indices)


# =============================================================================
# 3. INVERSION / OPTIMIZATION
# =============================================================================

def run_inversion(data):
    """
    Performs the optimization/solver loop for ptychographic reconstruction.
    
    This function sets up the model and optimizer, then runs the reconstruction loop
    which iteratively calls the forward_operator to generate predictions and updates
    the model parameters to minimize the loss.
    
    Args:
        data: Dictionary containing preprocessed data from load_and_preprocess_data
        
    Returns:
        dict containing:
            - model: Trained PtychoAD model
            - optimizer: Final optimizer state
            - output_path: Path where results are saved
            - solver_time: Total time for reconstruction
    """
    params = deepcopy(data['params'])
    device = data['device']
    logger = data['logger']
    init = data['init']
    loss_fn = data['loss_fn']
    constraint_fn = data['constraint_fn']
    verbose = data['verbose']
    
    start_t = time_sync()
    
    vprint("### Starting the PtyRADSolver in reconstruct mode ###")
    vprint(" ")
    
    # Create the model (which contains the forward operator)
    model = PtychoAD(init.init_variables, params['model_params'], device=device, verbose=verbose)
    
    # Create optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    
    # Prepare reconstruction indices, batches, and output path
    indices, batches, output_path = prepare_recon(model, init, params)
    
    # Flush logger if needed
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)
    
    # Run the reconstruction loop
    # Note: recon_loop internally calls model.forward() (i.e., forward_operator) 
    # during each iteration to compute predictions
    recon_loop(
        model, init, params, optimizer, loss_fn, constraint_fn,
        indices, batches, output_path, acc=None
    )
    
    end_t = time_sync()
    solver_t = end_t - start_t
    time_str = "" if solver_t < 60 else f", or {parse_sec_to_time_str(solver_t)}"
    
    vprint(f"### The PtyRADSolver is finished in {solver_t:.3f} sec{time_str} ###")
    vprint(" ")
    
    if logger is not None and logger.flush_file:
        logger.close()
    
    # End the process properly when in DDP mode
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return {
        'model': model,
        'optimizer': optimizer,
        'output_path': output_path,
        'solver_time': solver_t,
        'loss_iters': model.loss_iters,
        'iter_times': model.iter_times
    }


# =============================================================================
# 4. EVALUATION AND RESULTS
# =============================================================================

def evaluate_results(results):
    """
    Calculates metrics and summarizes reconstruction results.
    
    Args:
        results: Dictionary containing reconstruction results from run_inversion
        
    Returns:
        dict containing evaluation metrics and summary
    """
    model = results['model']
    output_path = results['output_path']
    solver_time = results['solver_time']
    loss_iters = results.get('loss_iters', [])
    
    vprint("### Reconstruction Results Summary ###")
    vprint(f"Output path: {output_path}")
    vprint(f"Total solver time: {solver_time:.3f} sec")
    
    # Get object and probe shapes
    obj_shape = model.opt_obja.shape
    probe_shape = model.get_complex_probe_view().shape
    
    vprint(f"Object shape: {obj_shape}")
    vprint(f"Probe shape: {probe_shape}")
    
    # Extract final loss if available
    if len(loss_iters) > 0:
        final_loss = loss_iters[-1]
        # Handle case where final_loss might be a tuple or list
        if isinstance(final_loss, (tuple, list)):
            final_loss_value = final_loss[0] if len(final_loss) > 0 else None
        else:
            final_loss_value = final_loss
        
        if final_loss_value is not None:
            vprint(f"Final loss: {float(final_loss_value):.6f}")
    
    # Calculate basic statistics on reconstructed object
    with torch.no_grad():
        obj_amp = model.opt_obja.detach()
        obj_phase = model.opt_objp.detach()
        
        obj_amp_mean = obj_amp.mean().item()
        obj_amp_std = obj_amp.std().item()
        obj_phase_mean = obj_phase.mean().item()
        obj_phase_std = obj_phase.std().item()
        
        vprint(f"Object amplitude - mean: {obj_amp_mean:.4f}, std: {obj_amp_std:.4f}")
        vprint(f"Object phase - mean: {obj_phase_mean:.4f}, std: {obj_phase_std:.4f}")
    
    return {
        'output_path': output_path,
        'solver_time': solver_time,
        'obj_shape': obj_shape,
        'probe_shape': probe_shape,
        'obj_amp_mean': obj_amp_mean,
        'obj_amp_std': obj_amp_std,
        'obj_phase_mean': obj_phase_mean,
        'obj_phase_std': obj_phase_std,
        'model': model
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Configuration
    params_path = "PSO_reconstruct.yml"
    gpuid = 0  # Set to None for CPU
    
    # Initialize logger
    logger = CustomLogger(
        log_file='ptyrad_log.txt',
        log_dir='auto',
        prefix_time='datetime',
        show_timestamp=True
    )
    
    # Step 1: Load and preprocess data
    data = load_and_preprocess_data(params_path, gpuid=gpuid, logger=logger)
    
    # Step 2: Run inversion (optimization)
    # Note: run_inversion internally uses forward_operator via model.forward()
    results = run_inversion(data)
    
    # Step 3: Evaluate results
    evaluation = evaluate_results(results)
    
    # Final confirmation
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")