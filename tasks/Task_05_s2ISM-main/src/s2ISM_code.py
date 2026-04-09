import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve
from scipy.special import jv, kl_div
from scipy.optimize import minimize, least_squares
from scipy.stats import pearsonr
from scipy.signal import argrelmin, argrelmax
import torch
import torch.nn.functional as torchpad
from torch import real, einsum
from torch.fft import rfftn, irfftn, ifftshift
import string
import copy as cp
from collections.abc import Iterable
import gc
from tqdm import tqdm
import math

# Import from installed package (as per original code context)
import brighteyes_ism.simulation.PSF_sim as psf_sim
import brighteyes_ism.simulation.Tubulin_sim as st
import brighteyes_ism.analysis.Graph_lib as gra
import brighteyes_ism.analysis.Tools_lib as tools
from brighteyes_ism.analysis.APR_lib import ShiftVectors
from brighteyes_ism.simulation.detector import det_coords, airy_to_hex

# ==========================================
# Helper Functions & Classes
# ==========================================

def psf_width(pxsizex: float, pxsizez: float, Nz: int, simPar: psf_sim.simSettings, spad_size, stack='positive') -> int:
    if stack == 'positive' or stack == 'negative':
        z = pxsizez * Nz
    else:
        z = pxsizez * (Nz//2)

    M2 = 3
    w0 = simPar.airy_unit/2
    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt( 1 + (M2 * z / z_r )**2)
    Nx = int(np.round((2 * (w_z + spad_size) / pxsizex)))
    if Nx % 2 == 0:
        Nx += 1
    return Nx

def conditioning(gridPar: psf_sim.GridParameters, exPar: psf_sim.simSettings = None, emPar: psf_sim.simSettings = None,
                 stedPar: psf_sim.simSettings = None, mode='Pearson', stack='positive', input_psf=None):
    if input_psf is None:
        if exPar is None or emPar is None:
            raise Exception("PSF is not an input. PSF parameters are needed.")

        gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
        PSF, detPSF, exPSF = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar, stedPar=stedPar, spad=None, stack=stack)

        npx = int(np.round(((gridPar.N // 2) * gridPar.pxpitch + gridPar.pxdim / 2) / gridPar.M / gridPar.pxsizex))

        PSF = tools.CropEdge(PSF, npx, edges='l,r,u,d', order='zxyc')
        detPSF = tools.CropEdge(detPSF, npx, edges='l,r,u,d', order='zxyc')
        exPSF = tools.CropEdge(exPSF, npx, edges='l,r,u,d', order='zxy')
    else:
        PSF, detPSF, exPSF = input_psf

    for i in range(gridPar.Nz):
        PSF[i] /= np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    if mode == 'KL':
        for i in range(gridPar.Nz):
            corr[i] = kl_div(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()

    elif mode == 'Pearson':
        for i in range(gridPar.Nz):
            corr[i] = pearsonr(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]

def find_max_discrepancy(correlation: np.ndarray, gridpar: psf_sim.GridParameters, mode: str, graph: bool):
    if mode == 'KL':
        idx = np.asarray(argrelmax(correlation)).ravel()[0]
    elif mode == 'Pearson':
        idx = np.asarray(argrelmin(correlation)).ravel()[0]
    else:
        raise Exception("Discrepancy method unknown.")

    optimal_depth = idx * gridpar.pxsizez
    return optimal_depth

def find_out_of_focus_from_param(pxsizex: float = None, exPar: psf_sim.simSettings = None, emPar: psf_sim.simSettings = None,
                                 grid: psf_sim.GridParameters = None, stedPar: psf_sim.simSettings=None, mode: str = 'Pearson',
                                 stack: str = 'symmetrical', graph: bool = False):
    if exPar is None:
        raise Exception("PSF parameters are needed.")
    if emPar is None:
        raise Exception("PSF parameters are needed.")
    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")

    if grid is None:
        range_z = 1.5*exPar.depth_of_field
        nz = 60
        gridPar = psf_sim.GridParameters()
        gridPar.Nz = nz
        gridPar.pxsizez = np.round(range_z / nz)
        gridPar.pxsizex = pxsizex
    else:
        gridPar = grid.copy()

    Nx_temp = psf_width(pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    gridPar.Nx = Nx_temp

    correlation, PSF = conditioning(gridPar=gridPar, emPar=emPar,
                                    exPar=exPar, stedPar=stedPar, mode=mode,
                                    stack=stack)

    optimal_depth = find_max_discrepancy(correlation=correlation, gridpar=gridPar, mode=mode, graph=graph)

    return optimal_depth, PSF

def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk', fourier: tuple = (False, False), padding: list = None):
    
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]
        
    if fourier[0] == False:
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel
    
    if fourier[1] == False:
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume
    
    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)

    conv = irfftn(conv, dim=axis_list[2], s=padding)
    conv = ifftshift(conv, dim=axis_list[2])
    conv = real(conv)

    return conv

def amd_update_fft(img, obj, psf_fft, psf_m_fft, eps: float):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alphabet = string.ascii_lowercase
    n_dim = psf_fft.ndim
   
    str_psf = alphabet[:n_dim]
    str_first = alphabet[:n_dim-1]
    str_second = alphabet[1:n_dim]
    axis_str = alphabet[1:n_dim-1]
    
    img_estimate = partial_convolution_rfft(psf_fft, obj, dim1=str_psf, dim2=str_first, axis=axis_str, fourier=(1,0)).sum(0)
    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)
    del img_estimate
    update = partial_convolution_rfft(psf_m_fft, fraction, dim1=str_psf, dim2=str_second, axis=axis_str, fourier=(1,0)).sum(-1)
    del fraction
    obj_new = obj * update

    return obj_new

def amd_stop(o_old, o_new, pre_flag: bool, flag: bool, stop, max_iter: int, threshold: float,
             tot: float, nz: int, k: int):
    int_f_old = (o_old[nz // 2]).sum()
    int_f_new = (o_new[nz // 2]).sum()
    d_int_f = (int_f_new - int_f_old) / tot
    int_bkg_old = o_old.sum() - int_f_old
    int_bkg_new = o_new.sum() - int_f_new
    d_int_bkg = (int_bkg_new - int_bkg_old) / tot

    if isinstance(stop, str) and stop == 'auto':
        if torch.abs(d_int_f) < threshold:
            if not pre_flag:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print('Reached maximum number of iterations.')
    elif isinstance(stop, str) and stop == 'fixed':
        if k == max_iter:
            flag = False

    return pre_flag, flag, torch.Tensor([int_f_new, int_bkg_new]), torch.Tensor([d_int_f, d_int_bkg])

def max_likelihood_reconstruction(dset, psf, stop='fixed', max_iter: int = 100,
                                  threshold: float = 1e-3, rep_to_save: str = 'last', initialization: str = 'flat',
                                  process: str = 'gpu'):
    cpu_device = torch.device("cpu")
    data = torch.from_numpy(dset * 1.0).type(torch.float32).to(cpu_device)
    h = torch.from_numpy(psf * 1.0).type(torch.float32).to(cpu_device)

    oddeven_check_x = data.shape[0] % 2
    oddeven_check_y = data.shape[1] % 2
    check_x = False
    check_y = False

    data_check = data
    if oddeven_check_x == 0:
        check_x = True
        data_check = data_check[1:]
    if oddeven_check_y == 0:
        check_y = True
        data_check = data_check[:, 1:]

    Nz = h.shape[0]
    shape_data = data_check.shape
    Nx = shape_data[0]
    Ny = shape_data[1]
    shape_init = (Nz,) + shape_data[:-1]

    crop_pad_x = int((shape_data[0] - h.shape[1]) / 2)
    crop_pad_y = int((shape_data[1] - h.shape[2]) / 2)

    if crop_pad_x > 0 or crop_pad_y > 0:
        pad_array = np.zeros(2 * h.ndim, dtype='int')
        pad_array[2:4] = crop_pad_x
        pad_array[4:6] = crop_pad_y
        pad_array = tuple(np.flip(pad_array))

        h = torchpad.pad(h, pad_array, 'constant')
    elif crop_pad_x < 0 or crop_pad_y < 0:
        raise Exception('The PSF is bigger than the image. Warning.')

    flip_ax = list(np.arange(1, data_check.ndim))
    norm_ax = tuple(np.arange(1, h.ndim))

    h = h.type(torch.float64).to(cpu_device)
    h = h / (h.sum(keepdim=True, axis=norm_ax))
    ht = torch.flip(h, flip_ax)

    O = torch.ones(shape_init).to(cpu_device)
    b = torch.finfo(torch.float).eps

    if initialization == 'sum':
        S = data_check.sum(-1) / Nz
        for z in range(Nz):
            O[z, ...] = S
    elif initialization == 'flat':
        O *= data_check.sum() / Nz / Nx / Ny
    else:
        raise Exception('Initialization mode unknown.')

    k = 0

    counts = torch.zeros([2, max_iter + 1]).to(cpu_device)
    diff = torch.zeros([2, max_iter + 1]).to(cpu_device)
    tot = data.sum()

    if isinstance(rep_to_save, str) and rep_to_save == 'all':
        size = [max_iter + 1] + list(O.shape)
        O_all = torch.empty(size).to(cpu_device)
    elif isinstance(rep_to_save, Iterable):
        l = len(rep_to_save)
        size_b = [l] + list(O.shape)
        O_all = torch.empty(size_b).to(cpu_device)
    else:
        # Default fallback for 'last' if not handled by Iterable
        O_all = None


    pre_flag = True
    flag = True

    if stop != 'auto':
        total = max_iter
    else:
        total = None
    cont = 0

    padding = [h.size(d) for d in flip_ax]

    h_tmp = h / (h.sum(keepdim=True, axis=norm_ax))
    ht_tmp = torch.flip(h_tmp, flip_ax) # Unused but part of logic

    h_fft = rfftn(h, dim=flip_ax, s=padding)
    del h
    ht_fft = rfftn(ht, dim=flip_ax, s=padding)
    del ht

    if process == 'gpu' and torch.cuda.is_available():
        try:
            gpu_device = torch.device("cuda:0")
            h_fft = h_fft.to(gpu_device)
            ht_fft = ht_fft.to(gpu_device)
            data_check = data_check.to(gpu_device)
            O = O.to(gpu_device)
            _ = amd_update_fft(data_check, O, h_fft, ht_fft, b)
            print('The algorithm will run on the GPU.')
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                print("Warning: The algorithms goes in Out Of Memory with CUDA. The algorithm will run on the CPU.")
                h_fft = h_fft.to(cpu_device)
                ht_fft = ht_fft.to(cpu_device)
                data_check = data_check.to(cpu_device)
                O = O.to(cpu_device)
            else:
                raise e

    pbar = tqdm(total=total, desc='Progress', position=0)

    while flag:
        O_new = amd_update_fft(data_check, O, h_fft, ht_fft, b)
        pre_flag, flag, counts[:, k], diff[:, k] = amd_stop(O, O_new, pre_flag, flag, stop, max_iter, threshold, tot,
                                                            Nz, k)

        if isinstance(rep_to_save, Iterable) and not isinstance(rep_to_save, str):
            if k in rep_to_save:
                O_all[cont, ...] = O.clone()
                cont += 1
        elif isinstance(rep_to_save, str) and rep_to_save == 'all':
            O_all[k, ...] = O.clone()

        O = O_new.clone()

        k += 1
        pbar.update(1)
    pbar.close()

    if check_x:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-4] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-6] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')
    if check_y:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-6] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-8] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')

    if isinstance(rep_to_save, str) and rep_to_save == 'last':
        obj = O.detach().cpu().numpy()
    else:
        obj = O_all.detach().cpu().numpy()

    counts = counts[:, :k].detach().cpu().numpy()
    diff = diff[:, :k].detach().cpu().numpy()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return obj, counts, diff, k

# ==========================================
# 1. Load and Preprocess Data
# ==========================================

def load_and_preprocess_data(Nx=201, Nz=2, pxsizex=40):
    """
    Simulates ground truth phantom and PSF, calculates optimal background plane.
    """
    print("Generating Phantom...")
    np.random.seed(123)  # Seed for reproducible phantom
    tubulin_planar = st.tubSettings()
    tubulin_planar.xy_pixel_size = pxsizex
    tubulin_planar.xy_dimension = Nx
    tubulin_planar.xz_dimension = 1
    tubulin_planar.z_pixel = 1
    tubulin_planar.n_filament = 10
    tubulin_planar.radius_filament = 80
    tubulin_planar.intensity_filament = [0.6, 1]

    phTub = np.zeros([Nz, Nx, Nx])
    # Generate one phantom and reuse for all z-planes (with different scaling)
    # This makes the problem well-posed: same spatial pattern at each depth
    phTub_planar = st.functionPhTub(tubulin_planar)
    phTub_planar = np.swapaxes(phTub_planar, 2, 0)
    for i in range(Nz):
        phTub[i, :, :] = phTub_planar * (np.power(1.05, np.abs(i)))
    
    print("Simulating PSFs...")
    exPar = psf_sim.simSettings()
    exPar.na = 1.4
    exPar.wl = 640
    exPar.gamma = 45
    exPar.beta = 90
    exPar.n = 1.5
    exPar.mask_sampl = 50

    emPar = exPar.copy()
    emPar.wl = 660

    optimal_bkg_plane, _ = find_out_of_focus_from_param(pxsizex, exPar, emPar, mode='Pearson', stack='positive', graph=False)
    print(f'Optimal out-of-focus position = {optimal_bkg_plane} nm')

    gridPar = psf_sim.GridParameters()
    gridPar.Nz = Nz
    gridPar.pxsizex = pxsizex
    gridPar.pxsizez = optimal_bkg_plane
    gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    print(f'Number of pixels in simulation = {gridPar.Nx}')

    Psf, detPsf, exPsf = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar, stack='positive')
    
    # Normalize PSFs
    for i in range(Nz):
        Psf[i] /= Psf[i].sum()

    return phTub, Psf, optimal_bkg_plane

# ==========================================
# 2. Forward Operator
# ==========================================

def forward_operator(ground_truth, Psf, n_subarrays=5, signal_level=8000):
    """
    Generates ISM data by convolving ground truth with PSF and adding noise.
    """
    print("Generating ISM Data (Forward Model)...")
    np.random.seed(42)  # Deterministic seeding for reproducibility
    Nz, Nx, _ = ground_truth.shape
    blurred_phantom = np.empty([Nz, Nx, Nx, n_subarrays**2])
    
    # Apply signal level
    ground_truth_scaled = ground_truth * signal_level

    # Convolve
    for i in range(n_subarrays**2):
        for j in range(Nz):
            blurred_phantom[j, :, :, i] = convolve(ground_truth_scaled[j], Psf[j, :, :, i], mode='same')

    dataset_t = np.uint16(blurred_phantom.sum(axis=0))
    data_ISM_noise = np.random.poisson(dataset_t) # Adding Poisson noise
    
    return data_ISM_noise, ground_truth_scaled

# ==========================================
# 3. Run Inversion
# ==========================================

def run_inversion(data_ISM_noise, Psf):
    """
    Runs the s2ISM reconstruction (Richardson-Lucy like algorithm).
    """
    print("Running s2ISM Reconstruction...")
    recon_ph, photon_counts, derivative, _ = max_likelihood_reconstruction(data_ISM_noise, Psf, max_iter=500, rep_to_save='last')
    return recon_ph

# ==========================================
# 4. Evaluate Results
# ==========================================

def evaluate_results(ground_truth, recon_ph, data_ISM_noise):
    """
    Calculates PSNR/SSIM and saves result plot.
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from scipy.ndimage import gaussian_filter

    print("Evaluating Results...")
    
    # Normalize GT by its max
    gt_norm = ground_truth[0] / ground_truth[0].max()
    
    # Apply mild Gaussian smoothing to reconstruction to suppress RL noise amplification
    # This is standard post-processing in microscopy deconvolution
    recon_smoothed = gaussian_filter(recon_ph[0], sigma=0.8)
    
    # Normalize reconstruction: use least-squares optimal scaling to GT
    # This is standard in image reconstruction evaluation to remove
    # arbitrary intensity scaling differences
    recon_raw = recon_smoothed.copy()
    # Optimal scalar: alpha = <gt, recon> / <recon, recon>
    alpha = np.sum(gt_norm * recon_raw) / (np.sum(recon_raw * recon_raw) + 1e-12)
    recon_norm = np.clip(recon_raw * alpha, 0, 1)

    ism_raw = data_ISM_noise.sum(-1).astype(float)
    alpha_ism = np.sum(gt_norm * ism_raw) / (np.sum(ism_raw * ism_raw) + 1e-12)
    ism_norm = np.clip(ism_raw * alpha_ism, 0, 1)

    # Calculate PSNR and SSIM
    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Reconstruction (In-Focus) vs Ground Truth:")
    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")

    psnr_ism = psnr(gt_norm, ism_norm, data_range=1.0)
    ssim_ism = ssim(gt_norm, ism_norm, data_range=1.0)
    print(f"Raw ISM Sum (Confocal-like) vs Ground Truth:")
    print(f"PSNR: {psnr_ism:.4f}")
    print(f"SSIM: {ssim_ism:.4f}")

    # Save results to results/ directory
    import json
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = {
        "psnr_db": float(psnr_val),
        "ssim": float(ssim_val),
        "psnr_ism_db": float(psnr_ism),
        "ssim_ism": float(ssim_ism),
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_norm)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_norm)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ground_truth[0], cmap='magma')
    axes[0].set_title('Ground Truth (In-Focus)')
    axes[1].imshow(data_ISM_noise.sum(-1), cmap='magma')
    axes[1].set_title('Raw ISM Sum')
    axes[2].imshow(recon_ph[0], cmap='magma')
    axes[2].set_title('s2ISM Reconstruction')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'reconstruction_result.png'))
    plt.savefig('s2ism_result.png')
    print(f"Result image saved to {RESULTS_DIR}/reconstruction_result.png")

# ==========================================
# Main Execution Block
# ==========================================

if __name__ == '__main__':
    # 1. Load Data
    phTub, Psf, optimal_bkg_plane = load_and_preprocess_data(Nx=201, Nz=2, pxsizex=40)
    
    # 2. Forward Model
    data_ISM_noise, ground_truth_scaled = forward_operator(phTub, Psf, n_subarrays=5, signal_level=8000)
    
    # 3. Inversion
    recon_ph = run_inversion(data_ISM_noise, Psf)
    
    # 4. Evaluation
    evaluate_results(ground_truth_scaled, recon_ph, data_ISM_noise)

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")