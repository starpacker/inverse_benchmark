import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from itertools import permutations

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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
