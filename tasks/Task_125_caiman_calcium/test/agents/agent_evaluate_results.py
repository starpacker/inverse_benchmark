import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter1d

def evaluate_results(
    true_spikes: np.ndarray,
    estimated_spikes: np.ndarray,
    fluorescence_noisy: np.ndarray,
    reconstructed_fluorescence: np.ndarray,
    spike_tol_frames: int = 2,
    results_dir: str = None
) -> dict:
    """
    Evaluate deconvolution results and create visualizations.
    
    Parameters
    ----------
    true_spikes : array of shape (n_neurons, n_frames)
        Ground truth spike trains
    estimated_spikes : array of shape (n_neurons, n_frames)
        Estimated spike trains
    fluorescence_noisy : array of shape (n_neurons, n_frames)
        Noisy fluorescence traces
    reconstructed_fluorescence : array of shape (n_neurons, n_frames)
        Reconstructed fluorescence traces
    spike_tol_frames : int
        Tolerance window for spike detection (±frames)
    results_dir : str
        Directory to save results (if None, uses default)
    
    Returns
    -------
    dict containing all computed metrics
    """
    
    def compute_psnr(reference, estimate):
        """Compute PSNR between reference and estimate signals."""
        mse = np.mean((reference - estimate) ** 2)
        if mse < 1e-12:
            return 100.0
        data_range = np.max(reference) - np.min(reference)
        return 20 * np.log10(data_range / np.sqrt(mse))
    
    def compute_correlation(a, b):
        """Compute Pearson correlation coefficient."""
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        num = np.sum(a_centered * b_centered)
        den = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
        if den < 1e-12:
            return 0.0
        return num / den
    
    def compute_spike_detection_metrics(true_spk, est_spk, tolerance=2):
        """Compute spike detection precision, recall, and F1 score."""
        true_locs = np.where(true_spk > 0.3)[0]
        est_locs = np.where(est_spk > np.max(est_spk) * 0.15)[0]
        
        if len(true_locs) == 0 and len(est_locs) == 0:
            return 1.0, 1.0, 1.0
        if len(true_locs) == 0:
            return 0.0, 1.0, 0.0
        if len(est_locs) == 0:
            return 1.0, 0.0, 0.0
        
        true_positives = 0
        matched_est = set()
        for t_loc in true_locs:
            for e_idx, e_loc in enumerate(est_locs):
                if abs(t_loc - e_loc) <= tolerance and e_idx not in matched_est:
                    true_positives += 1
                    matched_est.add(e_idx)
                    break
        
        precision = true_positives / len(est_locs) if len(est_locs) > 0 else 0.0
        recall = true_positives / len(true_locs) if len(true_locs) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        
        return precision, recall, f1
    
    n_neurons = true_spikes.shape[0]
    
    psnr_list = []
    cc_spike_list = []
    cc_fluor_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(n_neurons):
        # PSNR on fluorescence reconstruction
        psnr = compute_psnr(fluorescence_noisy[i], reconstructed_fluorescence[i])
        psnr_list.append(psnr)
        
        # Correlation on spike trains (smoothed for better comparison)
        true_smooth = gaussian_filter1d(true_spikes[i], sigma=3)
        est_smooth = gaussian_filter1d(estimated_spikes[i], sigma=3)
        cc_spk = compute_correlation(true_smooth, est_smooth)
        cc_spike_list.append(cc_spk)
        
        # Correlation on fluorescence
        cc_fl = compute_correlation(fluorescence_noisy[i], reconstructed_fluorescence[i])
        cc_fluor_list.append(cc_fl)
        
        # Spike detection metrics
        prec, rec, f1 = compute_spike_detection_metrics(
            true_spikes[i], estimated_spikes[i], tolerance=spike_tol_frames
        )
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
        print(f"  Neuron {i+1}: PSNR={psnr:.2f} dB, CC_spike={cc_spk:.4f}, "
              f"CC_fluor={cc_fl:.4f}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    
    mean_psnr = np.mean(psnr_list)
    mean_cc_spike = np.mean(cc_spike_list)
    mean_cc_fluor = np.mean(cc_fluor_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    
    print(f"\n  === Average Metrics ===")
    print(f"  PSNR (fluorescence): {mean_psnr:.2f} dB")
    print(f"  CC (spike trains):   {mean_cc_spike:.4f}")
    print(f"  CC (fluorescence):   {mean_cc_fluor:.4f}")
    print(f"  Spike Precision:     {mean_precision:.4f}")
    print(f"  Spike Recall:        {mean_recall:.4f}")
    print(f"  Spike F1 Score:      {mean_f1:.4f}")
    
    metrics = {
        'mean_psnr': mean_psnr,
        'mean_cc_spike': mean_cc_spike,
        'mean_cc_fluor': mean_cc_fluor,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'per_neuron_psnr': psnr_list,
        'per_neuron_cc_spike': cc_spike_list,
        'per_neuron_cc_fluor': cc_fluor_list,
        'per_neuron_precision': precision_list,
        'per_neuron_recall': recall_list,
        'per_neuron_f1': f1_list,
    }
    
    return metrics
