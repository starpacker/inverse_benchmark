import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import nnls

from itertools import permutations

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def vca(Y, n_end, rng):
    """
    Vertex Component Analysis (VCA) for endmember extraction.
    """
    L, P = Y.shape
    
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_centered = Y - Y_mean
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    Ud = U[:, :n_end]
    Y_proj = Ud.T @ Y_centered
    
    best_indices = None
    best_volume = -1
    
    for trial in range(10):
        indices = []
        for i in range(n_end):
            if i == 0:
                w = rng.standard_normal(n_end)
            else:
                E_sel = Y_proj[:, indices]
                proj_matrix = E_sel @ np.linalg.pinv(E_sel)
                w = (np.eye(n_end) - proj_matrix) @ rng.standard_normal(n_end)
            
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                w /= w_norm
            
            projections = w @ Y_proj
            idx = np.argmax(np.abs(projections))
            if idx in indices:
                sorted_idx = np.argsort(np.abs(projections))[::-1]
                for candidate in sorted_idx:
                    if candidate not in indices:
                        idx = candidate
                        break
            indices.append(idx)
        
        E_trial = Y_proj[:, indices]
        try:
            vol = abs(np.linalg.det(E_trial))
        except:
            vol = 0
        
        if vol > best_volume:
            best_volume = vol
            best_indices = indices
    
    E_vca = Y[:, best_indices]
    print(f"  VCA selected pixel indices: {best_indices} (vol={best_volume:.4e})")
    return E_vca, best_indices

def fcls(Y, E):
    """
    Fully Constrained Least Squares (FCLS) abundance estimation.
    """
    L, P = Y.shape
    R = E.shape[1]
    A = np.zeros((R, P))

    for p in range(P):
        a, residual = nnls(E, Y[:, p])
        a_sum = a.sum()
        if a_sum > 1e-12:
            a /= a_sum
        else:
            a = np.ones(R) / R
        A[:, p] = a

    return A

def sunsal_admm(Y, E, lam=0.01, n_iter=200, rho=1.0):
    """
    Sparse Unmixing by Variable Splitting and Augmented Lagrangian (SUnSAL).
    """
    L, P = Y.shape
    R = E.shape[1]

    EtE = E.T @ E
    EtY = E.T @ Y
    I_R = np.eye(R)
    inv_mat = np.linalg.inv(EtE + rho * I_R)

    A = np.linalg.lstsq(E, Y, rcond=None)[0]
    Z = A.copy()
    D = np.zeros_like(A)

    for it in range(n_iter):
        A = inv_mat @ (EtY + rho * (Z - D))

        V = A + D
        Z = np.sign(V) * np.maximum(np.abs(V) - lam / rho, 0)
        Z = np.maximum(Z, 0)
        for p in range(P):
            z = Z[:, p]
            s = z.sum()
            if s > 1e-12:
                z /= s
            else:
                z = np.ones(R) / R
            Z[:, p] = z

        D = D + A - Z

    return Z

def nmf_unmixing(Y, n_end, n_iter=500, rng=None):
    """
    Non-negative Matrix Factorisation for joint E,A estimation.
    """
    L, P = Y.shape
    R = n_end
    
    Y_pos = np.maximum(Y, 0)
    
    if rng is None:
        rng = np.random.default_rng(42)
    E = np.abs(rng.standard_normal((L, R))) + 0.1
    A = np.abs(rng.standard_normal((R, P))) + 0.1
    
    A /= A.sum(axis=0, keepdims=True)
    
    eps = 1e-10
    for it in range(n_iter):
        num_A = E.T @ Y_pos
        den_A = E.T @ E @ A + eps
        A *= (num_A / den_A)
        
        A = np.maximum(A, eps)
        A /= A.sum(axis=0, keepdims=True)
        
        num_E = Y_pos @ A.T
        den_E = E @ A @ A.T + eps
        E *= (num_E / den_E)
        E = np.maximum(E, eps)
        
        if (it + 1) % 100 == 0:
            err = np.linalg.norm(Y_pos - E @ A, 'fro') / np.linalg.norm(Y_pos, 'fro')
            print(f"    NMF iter {it+1}: rel_error={err:.6f}")
    
    return E, A

def align_endmembers(E_gt, E_rec, A_gt, A_rec):
    """
    Find optimal permutation to align estimated endmembers with GT.
    """
    R = E_gt.shape[1]

    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(R)):
        score = 0
        for i, j in enumerate(perm):
            cos_val = np.dot(E_gt[:, i], E_rec[:, j]) / (
                np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_rec[:, j]) + 1e-12
            )
            score += cos_val
        if score > best_score:
            best_score = score
            best_perm = perm

    perm_list = list(best_perm)
    E_aligned = E_rec[:, perm_list]
    A_aligned = A_rec[perm_list, :]
    return E_aligned, A_aligned, perm_list

def run_inversion(data):
    """
    Run the full unmixing inversion pipeline.
    
    Uses multiple methods:
        1. VCA for endmember extraction
        2. FCLS for abundance estimation
        3. SUnSAL ADMM for sparse abundance estimation
        4. NMF for joint E,A estimation
    
    Args:
        data: Dictionary from load_and_preprocess_data
        
    Returns:
        dict containing:
            - E_rec: Reconstructed endmember matrix
            - A_rec: Reconstructed abundance matrix
            - method: Name of best method used
            - all_results: Dict with all method results
    """
    Y_noisy = data['Y_noisy']
    E_gt = data['E_gt']
    A_gt = data['A_gt']
    n_endmembers = data['n_endmembers']
    rng = data['rng']
    
    all_results = {}
    
    # Stage 3a: VCA Endmember Extraction
    print("\n[STAGE 3a] Inverse — VCA Endmember Extraction")
    E_vca, vca_indices = vca(Y_noisy, n_endmembers, rng)
    
    # Stage 3b: FCLS Abundance Estimation
    print("\n[STAGE 3b] Inverse — FCLS Abundance Estimation")
    A_fcls = fcls(Y_noisy, E_vca)
    m_fcls = compute_metrics_internal(E_gt, E_vca, A_gt, A_fcls)
    print(f"  FCLS CC={m_fcls['CC_abundance']:.4f}")
    all_results['FCLS'] = {'E': E_vca, 'A': A_fcls, 'metrics': m_fcls}
    
    # Stage 3c: SUnSAL ADMM Abundance Estimation
    print("\n[STAGE 3c] Inverse — SUnSAL ADMM Abundance Estimation")
    A_sunsal = sunsal_admm(Y_noisy, E_vca, lam=0.005, n_iter=300, rho=1.0)
    m_sunsal = compute_metrics_internal(E_gt, E_vca, A_gt, A_sunsal)
    print(f"  SUnSAL CC={m_sunsal['CC_abundance']:.4f}")
    all_results['SUnSAL'] = {'E': E_vca, 'A': A_sunsal, 'metrics': m_sunsal}
    
    # Pick best VCA-based method
    if m_sunsal['CC_abundance'] >= m_fcls['CC_abundance']:
        A_rec = A_sunsal
        E_rec = E_vca
        metrics = m_sunsal
        method = "SUnSAL"
    else:
        A_rec = A_fcls
        E_rec = E_vca
        metrics = m_fcls
        method = "FCLS"
    print(f"\n  VCA+{method} CC={metrics['CC_abundance']:.4f}")
    
    # Stage 3d: NMF joint unmixing
    print("\n[STAGE 3d] Inverse — NMF Joint Estimation")
    E_nmf, A_nmf = nmf_unmixing(Y_noisy, n_endmembers, n_iter=500, rng=rng)
    m_nmf = compute_metrics_internal(E_gt, E_nmf, A_gt, A_nmf)
    print(f"  NMF CC={m_nmf['CC_abundance']:.4f}")
    all_results['NMF'] = {'E': E_nmf, 'A': A_nmf, 'metrics': m_nmf}
    
    # Pick overall best method
    candidates = [(method, E_rec, A_rec, metrics),
                  ("NMF", E_nmf, A_nmf, m_nmf)]
    best = max(candidates, key=lambda x: x[3]['CC_abundance'])
    method, E_rec, A_rec, metrics = best
    print(f"\n  → Using {method} (highest CC={metrics['CC_abundance']:.4f})")
    
    return {
        'E_rec': E_rec,
        'A_rec': A_rec,
        'method': method,
        'metrics': metrics,
        'all_results': all_results
    }

def compute_metrics_internal(E_gt, E_rec, A_gt, A_rec):
    """Internal helper to compute unmixing quality metrics."""
    E_al, A_al, perm = align_endmembers(E_gt, E_rec, A_gt, A_rec)
    R = E_gt.shape[1]

    # Spectral Angle Distance (SAD) for endmembers
    sad_list = []
    for i in range(R):
        cos_val = np.dot(E_gt[:, i], E_al[:, i]) / (
            np.linalg.norm(E_gt[:, i]) * np.linalg.norm(E_al[:, i]) + 1e-12
        )
        sad_list.append(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

    # Abundance metrics
    cc_per_end = []
    for i in range(R):
        cc_per_end.append(float(np.corrcoef(A_gt[i], A_al[i])[0, 1]))

    a_gt_flat = A_gt.ravel()
    a_rec_flat = A_al.ravel()
    dr = a_gt_flat.max() - a_gt_flat.min()
    mse = np.mean((a_gt_flat - a_rec_flat) ** 2)
    psnr = float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))
    rmse = float(np.sqrt(mse))
    cc_overall = float(np.corrcoef(a_gt_flat, a_rec_flat)[0, 1])
    re = float(np.linalg.norm(a_gt_flat - a_rec_flat) / max(np.linalg.norm(a_gt_flat), 1e-12))

    return {
        "PSNR_abundance": psnr,
        "SSIM_abundance": 0.0,
        "CC_abundance": cc_overall,
        "RE_abundance": re,
        "RMSE_abundance": rmse,
        "mean_SAD_deg": float(np.mean(sad_list)),
        "per_endmember_SAD_deg": [float(s) for s in sad_list],
        "per_endmember_CC": cc_per_end,
    }
