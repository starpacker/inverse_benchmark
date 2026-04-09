import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def align_and_compare(gt, recon):
    """
    Phase retrieval has ambiguities (translation, inversion).
    Try all 4 flips and pick best correlation.
    """
    best_cc = -1
    best_recon = recon.copy()

    candidates = [
        recon,
        np.flipud(recon),
        np.fliplr(recon),
        np.flipud(np.fliplr(recon)),
    ]

    for cand in candidates:
        # Try all circular shifts to find best alignment
        F_gt = np.fft.fft2(gt)
        F_cand = np.fft.fft2(cand)
        cross_corr = np.real(np.fft.ifft2(F_gt * np.conj(F_cand)))
        shift = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

        aligned = np.roll(np.roll(cand, shift[0], axis=0), shift[1], axis=1)

        # Compute CC
        gt_norm = gt - np.mean(gt)
        al_norm = aligned - np.mean(aligned)
        denom = np.sqrt(np.sum(gt_norm**2) * np.sum(al_norm**2))
        if denom > 0:
            cc = np.sum(gt_norm * al_norm) / denom
        else:
            cc = 0

        if cc > best_cc:
            best_cc = cc
            best_recon = aligned.copy()

    return best_recon, best_cc
