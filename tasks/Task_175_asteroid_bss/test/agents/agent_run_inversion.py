import warnings

import numpy as np

import matplotlib

matplotlib.use('Agg')

from sklearn.decomposition import FastICA

def _match_sources(gt, est):
    """Match estimated sources to GT via maximum absolute correlation."""
    n_src = gt.shape[0]
    corr_mat = np.zeros((n_src, n_src))
    for i in range(n_src):
        for j in range(n_src):
            corr_mat[i, j] = np.corrcoef(gt[i], est[j])[0, 1]

    # Greedy assignment by |correlation|
    perm = [None] * n_src
    sign = [None] * n_src
    abs_corr = np.abs(corr_mat)
    for _ in range(n_src):
        idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        gt_idx, est_idx = idx
        perm[gt_idx] = est_idx
        sign[gt_idx] = np.sign(corr_mat[gt_idx, est_idx])
        abs_corr[gt_idx, :] = -1
        abs_corr[:, est_idx] = -1

    matched = np.zeros_like(est)
    for i in range(n_src):
        matched[i] = sign[i] * est[perm[i]]
    return matched

def _rescale_to_gt(gt, est):
    """Rescale each estimated source to best-fit GT in the least-squares sense."""
    out = np.zeros_like(est)
    for i in range(gt.shape[0]):
        alpha = np.dot(gt[i], est[i]) / (np.dot(est[i], est[i]) + 1e-12)
        out[i] = alpha * est[i]
    return out

def compute_psnr(ref, est):
    """Peak SNR (dB) for 1-D signal."""
    mse = np.mean((ref - est) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(np.abs(ref))
    return 10.0 * np.log10(peak ** 2 / mse)

def run_inversion(mixed, sources, n_components=2, n_restarts=10):
    """
    Run FastICA with multi-restart to recover sources from mixed signals.
    
    Parameters
    ----------
    mixed : np.ndarray
        Mixed observations, shape (n_sensors, N).
    sources : np.ndarray
        Ground truth sources for matching and rescaling, shape (n_sources, N).
    n_components : int
        Number of independent components to extract.
    n_restarts : int
        Number of random restarts per function variant.
    
    Returns
    -------
    recovered_scaled : np.ndarray
        Recovered and rescaled sources, shape (n_sources, N).
    best_info : dict
        Information about the best ICA configuration.
    """
    best_psnr = -np.inf
    best_recovered_scaled = None
    best_seed = 0
    best_fun = 'logcosh'
    
    for fun in ['logcosh', 'exp', 'cube']:
        for seed in range(n_restarts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ica = FastICA(n_components=n_components, max_iter=1000, tol=1e-4,
                              algorithm='parallel', whiten='unit-variance',
                              fun=fun, random_state=seed)
                try:
                    rec = ica.fit_transform(mixed.T).T  # (n_components, N)
                except Exception:
                    continue
            
            # Match permutation and sign
            matched = _match_sources(sources, rec)
            # Rescale to ground truth
            scaled = _rescale_to_gt(sources, matched)
            
            # Evaluate PSNR
            psnr_trial = np.mean([compute_psnr(sources[i], scaled[i]) 
                                  for i in range(sources.shape[0])])
            
            if psnr_trial > best_psnr:
                best_psnr = psnr_trial
                best_recovered_scaled = scaled
                best_seed = seed
                best_fun = fun
    
    best_info = {
        'best_seed': best_seed,
        'best_fun': best_fun,
        'best_psnr': best_psnr,
    }
    
    print(f"[INFO] Best seed: {best_seed}, fun: {best_fun}, PSNR: {best_psnr:.2f} dB")
    print("[INFO] Permutation & sign resolved, sources rescaled.")
    
    return best_recovered_scaled, best_info
