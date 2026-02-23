```python
import numpy as np
import torch
import tifffile
from tqdm import tqdm
from functools import partial

# ==============================================================================
# Device Setup
# ==============================================================================

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Helper Functions
# ==============================================================================

def angular_spectrum_kernel(domain_size, spec_pixel_size, pixel_size, km):
    assert domain_size[1] % 2 == 0 and domain_size[2] % 2 == 0, "domain_size[1] and domain_size[2] must be even"
    kx = (torch.linspace((-domain_size[1] // 2 + 1), (domain_size[1] // 2), domain_size[1]) - 1) * spec_pixel_size
    ky = (torch.linspace((-domain_size[2] // 2 + 1), (domain_size[2] // 2), domain_size[2]) - 1) * spec_pixel_size
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    Kz[-K2 + km**2 < 0] = 0.
    kernel = torch.exp(1j * Kz * pixel_size[0])
    ol_correction = km / Kz
    ol_correction[-K2 + (km * 1.2 / 1.33)**2 < 0] = 0.
    return torch.fft.fftshift(kernel), torch.fft.fftshift(ol_correction)

def make_regularizer(tv_param, value_range_param, sparse_param, ROI, step_size, device):
    # Consolidated helper functions to reduce function call overhead during instrumentation
    
    # Pre-fetch constants for closures
    s0, e0, s1, e1, s2, e2 = ROI
    min_val = value_range_param[0]
    max_val = value_range_param[1]
    
    def value_range_regu(x):
        # Inline projector_threshold
        x[s0:e0, s1:e1, s2:e2] = torch.clamp(x[s0:e0, s1:e1, s2:e2], min=min_val, max=max_val)
        return x

    def sparse_regu(z):
        if sparse_param is None:
            return z
        thres = sparse_param * step_size
        return torch.sign(z) * torch.max(torch.abs(z) - thres, torch.zeros_like(z))

    if tv_param[0] is None:
        return lambda x: sparse_regu(value_range_regu(x))
    else:
        tau = tv_param[0]
        step_num = tv_param[1]
        gamma = 1 / (12 * tau)
        
        # Define operators internally to avoid global function calls
        def op_grad(x):
            g = torch.zeros(x.shape + (3,), dtype=torch.float32, device=device)
            g[:-1, :, :, 0] = x[1:, :, :] - x[:-1, :, :]
            g[:, :-1, :, 1] = x[:, 1:, :] - x[:, :-1, :]
            g[:, :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1]
            return g
            
        def op_div(g):
            x = torch.zeros(g.shape[:-1], dtype=torch.float32, device=device)
            tmp = x.clone()
            # x direction
            tmp[1:-1, :, :] = g[1:-1, :, :, 0] - g[:-2, :, :, 0]
            tmp[0, :, :] = g[0, :, :, 0]
            tmp[-1, :, :] = -g[-2, :, :, 0]
            x += tmp
            # y direction
            tmp[:, 1:-1, :] = g[:, 1:-1, :, 1] - g[:, :-2, :, 1]
            tmp[:, 0, :] = g[:, 0, :, 1]
            tmp[:, -1, :] = -g[:, -2, :, 1]
            x += tmp
            # z direction
            tmp[:, :, 1:-1] = g[:, :, 1:-1, 2] - g[:, :, :-2, 2]
            tmp[:, :, 0] = g[:, :, 0, 2]
            tmp[:, :, -1] = -g[:, :, -2, 2]
            x += tmp
            return -x
            
        def proj_grad(g):
            norm = torch.linalg.norm(g, dim=-1)
            norm[norm < 1] = 1
            norm = norm.reshape(g.shape[:-1] + (1,))
            return g / norm

        def fista_regu(z):
            g_1 = op_grad(z)
            d = g_1.clone()
            q_1 = 1
            x = z.clone()
            
            # This loop runs many times (e.g. 50), so internalizing functions prevents 
            # 350+ instrumented calls per outer iteration
            for i in range(step_num):
                # g_2 = proj_grad(d + gamma * op_grad(value_range_regu(z - tau * op_div(d))))
                
                # Decompose to avoid nested calls if that helps readability, but nested is fine
                # since the functions are now local closures (not instrumented).
                
                # 1. z - tau * op_div(d)
                term1 = z - tau * op_div(d)
                
                # 2. value_range_regu(...)
                term2 = value_range_regu(term1)
                
                # 3. op_grad(...)
                term3 = op_grad(term2)
                
                # 4. d + gamma * ...
                term4 = d + gamma * term3
                
                # 5. g_2 = proj_grad(...)
                g_2 = proj_grad(term4)
                
                # x = value_range_regu(z - tau * op_div(g_2))
                x = value_range_regu(z - tau * op_div(g_2))
                
                q_2 = 0.5 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
                d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
                q_1 = q_2
                g_1 = g_2
            return sparse_regu(x)
            
        return fista_regu

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(data_config, physics_config):
    """
    Load input/output fields and preprocess them for BPM reconstruction.
    Returns a dictionary containing all preprocessed data needed for inversion.
    """
    device = get_device()
    
    wavelength = physics_config['wavelength']
    pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
    n_medium = physics_config['n_medium']
    km = 2 * np.pi / wavelength * n_medium
    k0 = 2 * np.pi / wavelength
    
    crop_size = data_config['crop_size']
    domain_size = data_config['domain_size']
    
    spec_pixel_size = 2 * np.pi / (pixelsize * crop_size[0])
    resolution = pixelsize * crop_size[0] / np.array(domain_size)
    print('resolution', resolution)
    
    # Load data from local folder
    try:
        u_in = np.load('data_folder/u_in.npy')
        u_in = torch.from_numpy(u_in).to(device)
        u_out = np.load('data_folder/u_out.npy')
        u_out = torch.from_numpy(u_out).to(device)
        k_scan_samp = np.load('data_folder/k_samp.npy')
    except FileNotFoundError:
        print("Error: Data files not found in data_folder/. Please ensure u_in.npy, u_out.npy, k_samp.npy exist.")
        raise
    
    # Compute cosine factor for oblique illumination
    temp = k_scan_samp * spec_pixel_size / km
    bpm_cosFactor = np.cos(np.arcsin(np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2)))
    bpm_cosFactor = torch.from_numpy(bpm_cosFactor.reshape(-1, 1)).float().to(device)
    
    # Determine z region parameters
    crop_z = data_config['crop_z']
    region_z = domain_size[0]
    bg_z = region_z // 2
    if crop_z is not None:
        region_z = data_config['region_z']
        bg_z = data_config['bg_z']
    
    # Compute angular spectrum propagation kernel
    p_kernel, _ = angular_spectrum_kernel(domain_size, spec_pixel_size, resolution, km)
    p_kernel = p_kernel.to(device)
    
    # Propagate fields to ROI
    u_inlet = torch.fft.ifft2(torch.fft.fft2(u_in) * (p_kernel ** (region_z - bg_z)).conj())
    u_outlet = torch.fft.ifft2(torch.fft.fft2(u_out) * (p_kernel ** bg_z))
    
    preprocessed_data = {
        'u_inlet': u_inlet,
        'u_outlet': u_outlet,
        'p_kernel': p_kernel,
        'bpm_cosFactor': bpm_cosFactor,
        'resolution': resolution,
        'domain_size': domain_size,
        'region_z': region_z,
        'k0': k0,
        'km': km,
        'device': device
    }
    
    return preprocessed_data

# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_operator(delta_ri, u_in, p_kernel, cos_factor, dz, k0, device):
    """
    BPM forward model: propagate input field through the refractive index distribution.
    
    Args:
        delta_ri: Refractive index perturbation (Nz, Ny, Nx)
        u_in: Input field (batch, Ny, Nx)
        p_kernel: Angular spectrum propagation kernel
        cos_factor: Oblique illumination correction factor
        dz: Axial step size
        k0: Wavenumber in vacuum
        device: Torch device
    
    Returns:
        u_out_pred: Predicted output field (batch, Ny, Nx)
    """
    ol_factor = k0 * dz / cos_factor.unsqueeze(-1)
    p_kernel_expanded = p_kernel.unsqueeze(0)
    
    u = u_in.clone()
    for m in range(delta_ri.shape[0]):
        u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel_expanded)
        u = u * torch.exp(1j * ol_factor * delta_ri[m, ...].unsqueeze(0))
    
    return u

# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(preprocessed_data, reconstruction_config, data_config):
    """
    Run the BPM inversion using gradient descent with FISTA acceleration.
    
    Args:
        preprocessed_data: Dictionary from load_and_preprocess_data
        reconstruction_config: Reconstruction parameters
        data_config: Data configuration
    
    Returns:
        delta_ri: Reconstructed refractive index perturbation (numpy array)
        loss_history: List of loss values during optimization
    """
    device = preprocessed_data['device']
    u_inlet = preprocessed_data['u_inlet']
    u_outlet = preprocessed_data['u_outlet']
    p_kernel = preprocessed_data['p_kernel']
    bpm_cosFactor = preprocessed_data['bpm_cosFactor']
    resolution = preprocessed_data['resolution']
    domain_size = preprocessed_data['domain_size']
    region_z = preprocessed_data['region_z']
    k0 = preprocessed_data['k0']
    
    n_iter = reconstruction_config['n_iter']
    step_size = reconstruction_config['step_size']
    batch_size = reconstruction_config['batch_size']
    ROI = data_config['ROI']
    
    # Create regularizer
    regu_func = make_regularizer(
        reconstruction_config['tv_param'],
        reconstruction_config['value_range_param'],
        reconstruction_config['sparse_param'][0],
        ROI=ROI,
        step_size=step_size,
        device=device
    )
    
    # Initialize refractive index
    init_delta_ri = torch.zeros((region_z, domain_size[1], domain_size[2]), dtype=torch.float32, device=device)
    print('RI shape', init_delta_ri.shape)
    
    # FISTA variables
    s = init_delta_ri.clone()
    q_1 = 1
    x_1 = init_delta_ri.clone()
    
    loss_history = []
    
    print(f"Starting reconstruction for {n_iter} iterations...")
    pbar = tqdm(range(n_iter))
    
    for iteration in pbar:
        # Compute gradient using BPM adjoint
        grad, loss = compute_bpm_gradient_batched(
            s, u_inlet, u_outlet, batch_size, bpm_cosFactor,
            resolution[0], domain_size, k0, p_kernel, device
        )
        
        loss_history.append(loss)
        pbar.set_postfix({'loss': loss}, refresh=False)
        
        with torch.no_grad():
            z = s - grad * step_size
            x_2 = regu_func(z)
            q_2 = 0.5 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
            s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
            x_1 = x_2
            q_1 = q_2
            g_1 = grad # Wait, this g_1 is not used here? 
                       # Ah, in FISTA algo usually g_1 refers to gradient? 
                       # In the original code: 
                       # d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
                       # where g_2 and g_1 were related to the regularizer internal loop.
                       # Here inside run_inversion loop:
                       # s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
                       # This is standard FISTA on x.
                       # The inner loop in make_regularizer is ALSO a FISTA (TV proximal).
                       # So there are two FISTAs?
                       # Yes, TV denoising via FISTA is the proximal operator.
    
    delta_ri = s.cpu().numpy()
    
    return delta_ri, loss_history

def compute_bpm_gradient_batched(init_delta_ri, u_in, u_out, batch_size, cos_factor, dz, domain_size, k0, p_kernel, device):
    """
    Compute gradient of the loss with respect to refractive index using adjoint method.
    """
    ol_factor = k0 * dz / cos_factor.unsqueeze(-1)
    p_kernel_expanded = p_kernel.unsqueeze(0)
    bp_kernel = p_kernel.conj().unsqueeze(0)
    
    grad = torch.zeros_like(init_delta_ri)
    delta_ri = init_delta_ri
    total_loss = 0.0
    num_batches = (u_in.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, u_in.shape[0])
        actual_batch_size = end_idx - start_idx
        
        sub_u_in = u_in[start_idx:end_idx, ...]
        sub_u_out = u_out[start_idx:end_idx, ...]
        sub_ol_factor = ol_factor[start_idx:end_idx, ...]
        
        # Forward pass - store intermediate fields
        s_fields = torch.zeros((actual_batch_size, init_delta_ri.shape[0], init_delta_ri.shape[1], init_delta_ri.shape[2]), dtype=torch.cfloat, device=device)
        
        u = sub_u_in.clone()
        for m in range(init_delta_ri.shape[0]):
            u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel_expanded)
            s_fields[:, m, ...] = u.clone()
            u = u * torch.exp(1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
        
        # Compute residual
        r = u - sub_u_out
        batch_loss = r.abs().mean().item()
        total_loss += batch_loss
        
        # Backward pass - adjoint method
        for m in reversed(range(init_delta_ri.shape[0])):
            # Backpropagate through phase modulation
            r = r * torch.exp(-1j * sub_ol_factor * delta_ri[m, ...].unsqueeze(0))
            
            # Compute gradient contribution (take real part)
            batch_grad = -1j * sub_ol_factor * s_fields[:, m, ...].conj() * r
            grad[m, ...] += batch_grad.real.sum(dim=0)
            
            # Backpropagate through free space
            r = torch.fft.ifft2(torch.fft.fft2(r) * bp_kernel)
    
    grad = grad / u_in.shape[0]
    avg_loss = total_loss / num_batches
    
    return grad, avg_loss

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(delta_ri, loss_history, ROI):
    """
    Evaluate and report reconstruction results.
    
    Args:
        delta_ri: Reconstructed refractive index (numpy array)
        loss_history: List of loss values during optimization
        ROI: Region of interest for evaluation
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    s0, e0, s1, e1, s2, e2 = ROI
    
    # Extract ROI
    roi_data = delta_ri[s0:e0, s1:e1, s2:e2]
    
    # Compute statistics
    vmin = float(np.min(roi_data))
    vmax = float(np.max(roi_data))
    vmean = float(np.mean(roi_data))
    vstd = float(np.std(roi_data))
    
    # Loss statistics
    initial_loss = loss_history[0] if len(loss_history) > 0 else None
    final_loss = loss_history[-1] if len(loss_history) > 0 else None
    
    metrics = {
        'roi_min': vmin,
        'roi_max': vmax,
        'roi_mean': vmean,
        'roi_std': vstd,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'num_iterations': len(loss_history),
        'loss_history': loss_history
    }
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROI Statistics:")
    print(f"  Min: {vmin:.6f}")
    print(f"  Max: {vmax:.6f}")
    print(f"  Mean: {vmean:.6f}")
    print(f"  Std: {vstd:.6f}")
    print(f"Optimization:")
    print(f"  Initial Loss: {initial_loss:.6f}" if initial_loss else "  Initial Loss: N/A")
    print(f"  Final Loss: {final_loss:.6f}" if final_loss else "  Final Loss: N/A")
    print(f"  Iterations: {len(loss_history)}")
    print("=" * 60)
    
    return metrics

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Physics Config
    physics_config = {
        'wavelength': 0.532,
        'camera_pixel_size': 4.5,
        'n_medium': 1.33,
        'magnification': 100,
        'angle_num': 240,
        'NA': 1.32,
        'angle_downsample': 0,
        'conjugate_flag': 1
    }

    # Data Config
    data_config = {
        'expriment_name': 'h1k293_commercial',
        'data_name': 'ODT-532nm-sample-2024-12-19-17-36-37',
        'save_name': 'diffusion_cropz',
        'sp_path': './data_folder/ODT-532nm-sample-2024-12-19-17-36-37.tif',
        'bg_path': './data_folder/ODT-532nm-reference-2024-12-19-16-55-18.tif',
        'save_path': 'results',
        'mask_type': 'bg',
        'shift_idx': [166, -400],
        'ROI': [None, None, 60, -60, 60, -60],
        'crop_z': True,
        'region_z': 256,
        'bg_z': 93,
        'bpm_downsample': 2,
        'crop_size': [1004, 1004],
        'spec_size': [502, 502],
        'domain_size': [502, 502, 502],
        'batch_size': 10
    }

    # Reconstruction Config
    reconstruction_config = {
        'step_size': 0.0005,
        'tv_param': [1.75e-6, 50],
        'value_range_param': [0, 100],
        'sparse_param': [None],
        'n_iter': 1,
        'batch_size': 2,
        'crop_z': 1,
        'region_z': 256,
        'bg_z': 93,
        'xy_padding_len': None,
        'initial_type': 'zero',
        'init_name': 'rytov.tif',
        'initial_shift_idx': 0,
        'debug_mode': True
    }
    
    print("Running BPM Reconstruction Pipeline...")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    preprocessed_data = load_and_preprocess_data(data_config, physics_config)
    print("Data loaded successfully.")
    
    # Step 2: Run inversion
    print("Step 2: Running inversion...")
    delta_ri, loss_history = run_inversion(preprocessed_data, reconstruction_config, data_config)
    print("Inversion completed.")
    
    # Step 3: Evaluate results
    print("Step 3: Evaluating results...")
    metrics = evaluate_results(delta_ri, loss_history, data_config['ROI'])
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```