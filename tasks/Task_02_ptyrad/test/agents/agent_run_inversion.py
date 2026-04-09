import torch
import torch.nn as nn
from math import prod
import time
from torchvision.transforms.functional import gaussian_blur
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint, 
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.reconstruction import (
    create_optimizer, prepare_recon, parse_sec_to_time_str, 
    recon_loop, IndicesDataset
)
from ptyrad.forward import multislice_forward_model_vec_all

class PtychoAD(torch.nn.Module):
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params.get('detector_blur_std', 0)
            self.obj_preblur_std = model_params.get('obj_preblur_std', 0)
            
            self.loss_iters = []
            self.iter_times = []
            self.dz_iters = []
            self.avg_tilt_iters = []

            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = {}
            self.end_iter = {}
            self.lr_params = {}
            for key, p in model_params['update_params'].items():
                self.start_iter[key] = p.get('start_iter')
                self.end_iter[key] = p.get('end_iter')
                self.lr_params[key] = p['lr']

            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))

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

            self._create_grids()
            self._init_propagator_vars()
            
            self.optimizable_tensors = {
                'obja': self.opt_obja, 'objp': self.opt_objp,
                'obj_tilts': self.opt_obj_tilts, 'slice_thickness': self.opt_slice_thickness,
                'probe': self.opt_probe, 'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self._setup_optimizer_params()
            self._init_compilation_iters()

    def get_complex_probe_view(self):
        return torch.view_as_complex(self.opt_probe)

    def _create_grids(self):
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:]
        Noy, Nox = self.opt_objp.shape[-2:]

        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0)

        rpy, rpx = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=device), 
                                  torch.arange(Npx, dtype=torch.int32, device=device), indexing='ij')
        self.rpy_grid = rpy
        self.rpx_grid = rpx

        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(Npy, dtype=torch.float32, device=device),
                                  torch.fft.fftfreq(Npx, dtype=torch.float32, device=device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0)

    def _setup_optimizer_params(self):
        self.optimizable_params = []
        for param_name, lr in self.lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"Unknown param: {param_name}")
            self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})

    def _init_propagator_vars(self):
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y_full = self.opt_obj_tilts[:,0,None,None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:,1,None,None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full)))
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2)

    def _init_compilation_iters(self):
        compilation_iters = {1}
        for param_name in self.optimizable_tensors.keys():
            s = self.start_iter.get(param_name)
            e = self.end_iter.get(param_name)
            if s and s >= 1: compilation_iters.add(s)
            if e and e >= 1: compilation_iters.add(e)
        self.compilation_iters = sorted(compilation_iters)

    def get_obj_patches(self, indices):
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        object_roi = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_roi
        else:
            obj = object_roi.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            return gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)

    def get_probes(self, indices):
        probe = self.get_complex_probe_view()
        if self.shift_probes:
            return imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        return torch.broadcast_to(probe, (indices.shape[0], *probe.shape))

    def get_propagators(self, indices):
        tilt_obj = self.tilt_obj
        global_tilt = (self.opt_obj_tilts.shape[0] == 1)
        change_tilt = (self.lr_params['obj_tilts'] != 0)
        change_thickness = self.change_thickness
        
        dz = self.opt_slice_thickness
        Kz = self.Kz
        Ky, Kx = self.propagator_grid
        
        if global_tilt: tilts = self.opt_obj_tilts 
        else: tilts = self.opt_obj_tilts[indices] 
        tilts_y = tilts[:,0,None,None] / 1e3
        tilts_x = tilts[:,1,None,None] / 1e3
                
        if tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif tilt_obj and not change_thickness:
            if change_tilt:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness: 
            return torch_phasor(dz * Kz)[None,]
        else: 
            return self.H[None,]

    def get_measurements(self, indices=None):
        if indices is None: return self.measurements
        
        measurements = self.measurements[indices]
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros((measurements.shape[0], *self.meas_padded.shape[-2:]), dtype=measurements.dtype, device=self.device)
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas
            
        if self.meas_scale_factors is not None:
            scale = tuple(self.meas_scale_factors)
            if any(f != 1 for f in scale):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale, mode='bilinear')[0]
                measurements = measurements / prod(scale)
        return measurements

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, indices):
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
            
        self._current_object_patches = object_patches
        return dp_fwd

def _find_init_variables_robust(ctx):
    if 'init_variables' in ctx: return ctx['init_variables']
    
    if 'obj' in ctx and 'probe' in ctx: return ctx
    
    if 'params' in ctx and isinstance(ctx['params'], dict):
        if 'init_variables' in ctx['params']: return ctx['params']['init_variables']

    potential_keys = ['initializer', 'init']
    for key in potential_keys:
        if key in ctx:
            init_obj = ctx[key]
            if isinstance(init_obj, dict):
                if 'init_variables' in init_obj: return init_obj['init_variables']
                if 'obj' in init_obj: return init_obj
            else:
                if hasattr(init_obj, 'init_variables'): return init_obj.init_variables
                if hasattr(init_obj, 'variables'): return init_obj.variables
                if hasattr(init_obj, 'obj') and hasattr(init_obj, 'probe'): 
                    return {k: getattr(init_obj, k) for k in dir(init_obj) 
                            if not k.startswith('__') and not callable(getattr(init_obj, k))}

    for k, v in ctx.items():
        if isinstance(v, dict) and 'obj' in v and 'probe' in v:
            return v
        if not isinstance(v, dict) and hasattr(v, 'obj') and hasattr(v, 'probe'):
             return {attr: getattr(v, attr) for attr in dir(v) 
                     if not attr.startswith('__') and not callable(getattr(v, attr))}
            
    return None

def run_inversion(data_context):
    params = data_context.get('params')
    device = data_context.get('device')
    
    init_variables = _find_init_variables_robust(data_context)
                
    if init_variables is None:
        keys_present = list(data_context.keys())
        raise KeyError(f"Could not find 'init_variables' (or 'obj'/'probe') in data_context. Keys found: {keys_present}")

    initializer = data_context.get('initializer')
    if initializer is None:
        initializer = data_context.get('init')

    loss_fn = data_context.get('loss_fn')
    constraint_fn = data_context.get('constraint_fn')
    accelerator = data_context.get('accelerator')
    logger = data_context.get('logger')

    vprint("### Starting Reconstruction ###")

    model = PtychoAD(init_variables, params['model_params'], device=device, verbose=True)
    
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

    use_acc_device = (device is None and accelerator is not None)
    
    if not use_acc_device:
        indices, batches, output_path = prepare_recon(model, initializer, params)
    else:
        if params['model_params']['optimizer_params']['name'] == 'LBFGS' and accelerator.num_processes > 1:
            vprint("WARNING: LBFGS not supported for multi-GPU. Switching to Adam.")
            params['model_params']['optimizer_params']['name'] = 'Adam'
            model.optimizer_params['name'] = 'Adam'
            optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
        
        params['recon_params']['GROUP_MODE'] = 'random'
        indices, batches, output_path = prepare_recon(model, initializer, params)
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)

    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    t0 = time.time()
    recon_loop(
        model, initializer, params, optimizer, 
        loss_fn, constraint_fn, indices, batches, 
        output_path, acc=accelerator
    )
    solver_time = time.time() - t0

    return {
        "model": model,
        "optimizer": optimizer,
        "output_path": output_path,
        "logger": logger,
        "solver_time": solver_time
    }