import matplotlib

matplotlib.use('Agg')

import sys

import os

import warnings

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from skimage.restoration import denoise_tv_chambolle

from scipy.sparse.linalg import LinearOperator, cg as scipy_cg

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')

sys.path.insert(0, REPO_DIR)

warnings.filterwarnings('ignore', message='Samples will be rescaled')

def run_inversion(data, cg_iterations=500, cg_lambda=1e-3, tv_weights=None):
    """
    Run the inversion/reconstruction algorithms.
    
    Performs:
    1. Density-compensated adjoint (gridding) reconstruction
    2. Conjugate Gradient reconstruction on normal equations
    3. TV denoising sweep to find best result
    
    Parameters
    ----------
    data : dict
        Output from load_and_preprocess_data
    cg_iterations : int
        Maximum number of CG iterations
    cg_lambda : float
        Tikhonov regularization parameter
    tv_weights : list of float, optional
        TV weights to sweep (default: predefined list)
        
    Returns
    -------
    dict containing:
        - recon_adjoint: DC adjoint reconstruction
        - recon_cg: CG reconstruction
        - recon_final: best reconstruction (after TV if applicable)
        - best_method: string indicating best method
        - best_tv_label: label of best TV configuration
        - metrics_adjoint: (psnr, ssim, rmse) for adjoint
        - metrics_cg: (psnr, ssim, rmse) for CG
        - metrics_final: (psnr, ssim, rmse) for final
    """
    if tv_weights is None:
        tv_weights = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1]
    
    phantom = data['phantom']
    kdata = data['kdata']
    op_dc = data['op_dc']
    op_plain = data['op_plain']
    N = data['params']['N']
    
    # Helper: normalize to [0, 1]
    def normalize_to_01(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    # Helper: compute metrics
    def compute_metrics(gt, recon):
        gt_norm = normalize_to_01(gt)
        recon_norm = normalize_to_01(recon)
        psnr = peak_signal_noise_ratio(gt_norm, recon_norm, data_range=1.0)
        ssim = structural_similarity(gt_norm, recon_norm, data_range=1.0)
        rmse = np.sqrt(np.mean((gt_norm - recon_norm) ** 2))
        return psnr, ssim, rmse
    
    # 1. Density-compensated adjoint reconstruction (gridding)
    print("\n  Running density-compensated adjoint (gridding) reconstruction...")
    recon_adjoint = np.real(op_dc.adj_op(kdata))
    psnr_adj, ssim_adj, rmse_adj = compute_metrics(phantom, recon_adjoint)
    print(f"    Adjoint PSNR: {psnr_adj:.2f} dB, SSIM: {ssim_adj:.4f}")
    
    # 2. CG iterative reconstruction
    print(f"\n  Running CG reconstruction ({cg_iterations} max iterations, lambda={cg_lambda})...")
    
    def normal_eq(x_flat):
        x_img = x_flat.reshape(N, N).astype(np.complex64)
        result = op_plain.adj_op(op_plain.op(x_img)).flatten()
        if cg_lambda > 0:
            result += cg_lambda * x_flat
        return result
    
    AHA = LinearOperator((N * N, N * N), matvec=normal_eq, dtype=np.complex64)
    rhs = op_plain.adj_op(kdata).flatten()
    
    x_cg, info = scipy_cg(AHA, rhs, maxiter=cg_iterations, rtol=1e-8)
    
    if info == 0:
        print(f"    CG converged in < {cg_iterations} iterations")
    else:
        print(f"    CG reached max iterations ({cg_iterations}), info={info}")
    
    recon_cg = np.real(x_cg.reshape(N, N))
    psnr_cg, ssim_cg, rmse_cg = compute_metrics(phantom, recon_cg)
    print(f"    CG PSNR: {psnr_cg:.2f} dB, SSIM: {ssim_cg:.4f}")
    
    # 3. TV denoising sweep on both adjoint and CG bases
    print(f"\n  Sweeping TV weights on adjoint and CG bases...")
    gt_n = normalize_to_01(phantom)
    best_tv_psnr = -np.inf
    best_tv_recon = None
    best_tv_ssim = 0
    best_tv_rmse = 1
    best_tv_label = ''
    
    for base_label, base_img in [('adjoint', recon_adjoint), ('CG', recon_cg)]:
        base_norm = normalize_to_01(base_img)
        for tw in tv_weights:
            tv_img = denoise_tv_chambolle(base_norm, weight=tw)
            p = peak_signal_noise_ratio(gt_n, tv_img, data_range=1.0)
            s = structural_similarity(gt_n, tv_img, data_range=1.0)
            r = np.sqrt(np.mean((gt_n - tv_img) ** 2))
            if p > best_tv_psnr:
                best_tv_psnr = p
                best_tv_recon = tv_img
                best_tv_ssim = s
                best_tv_rmse = r
                best_tv_label = f'{base_label}+TV(w={tw})'
    
    print(f"    Best TV result: {best_tv_label}, PSNR={best_tv_psnr:.2f}, SSIM={best_tv_ssim:.4f}")
    
    # Determine best method overall
    methods = {
        'adjoint': (psnr_adj, ssim_adj, rmse_adj, recon_adjoint),
        'cg': (psnr_cg, ssim_cg, rmse_cg, recon_cg),
        'tv': (best_tv_psnr, best_tv_ssim, best_tv_rmse, best_tv_recon),
    }
    
    best_method = max(methods.keys(), key=lambda k: methods[k][0])
    final_psnr, final_ssim, final_rmse, final_recon = methods[best_method]
    
    print(f"\n  Best method: {best_method} (PSNR={final_psnr:.2f}, SSIM={final_ssim:.4f})")
    
    return {
        'recon_adjoint': recon_adjoint,
        'recon_cg': recon_cg,
        'recon_final': final_recon,
        'best_method': best_method,
        'best_tv_label': best_tv_label,
        'metrics_adjoint': (psnr_adj, ssim_adj, rmse_adj),
        'metrics_cg': (psnr_cg, ssim_cg, rmse_cg),
        'metrics_tv': (best_tv_psnr, best_tv_ssim, best_tv_rmse),
        'metrics_final': (final_psnr, final_ssim, final_rmse),
    }
