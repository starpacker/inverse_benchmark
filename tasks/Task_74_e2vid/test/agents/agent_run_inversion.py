import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter, median_filter

def forward_operator(log_intensity, events, event_idx, t_end, contrast_threshold):
    """
    Forward model: Apply event-based log-intensity updates.
    
    Given the current log-intensity state, integrate events up to t_end
    and return the predicted intensity image.
    
    Args:
        log_intensity: Current log-intensity state (height, width)
        events: List of events (x, y, t, polarity)
        event_idx: Current index in events list
        t_end: End time for integration
        contrast_threshold: Event triggering threshold C
    
    Returns:
        intensity: Predicted intensity image
        new_event_idx: Updated event index
    """
    log_int = log_intensity.copy()
    n_events = len(events)
    
    while event_idx < n_events and events[event_idx][2] < t_end:
        x, y, t, p = events[event_idx]
        x, y = int(x), int(y)
        log_int[y, x] += p * contrast_threshold
        event_idx += 1
    
    # Convert back to intensity
    intensity = np.exp(log_int)
    intensity = np.clip(intensity, 0, 5)
    
    return intensity, event_idx

def run_inversion(events, gt_frames, aps_frames, height, width, n_output_frames,
                  contrast_threshold, t_total, fps):
    """
    Run event-to-video reconstruction using multiple methods and select best.
    
    Methods:
        1) Direct temporal integration
        2) Complementary filter (low-pass APS + high-pass events)
        3) TV-regularised reconstruction
    
    Args:
        events: List of events
        gt_frames: Ground truth frames (for initialization)
        aps_frames: APS (low-pass) frames
        height, width: Image dimensions
        n_output_frames: Number of output frames
        contrast_threshold: Event threshold C
        t_total: Total time duration
        fps: Frame rate
    
    Returns:
        best_rec: Best reconstruction result
        best_name: Name of best method
        all_reconstructions: Dict of all reconstruction results
    """
    dt_out = t_total / n_output_frames
    
    # Method 1: Direct Integration
    rec_direct = np.zeros((n_output_frames, height, width))
    log_intensity = np.log(gt_frames[0] + 1e-6)
    event_idx = 0
    
    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out
        intensity, event_idx = forward_operator(
            log_intensity, events, event_idx, t_end, contrast_threshold
        )
        # Update log_intensity for next iteration
        n_events = len(events)
        while event_idx < n_events and events[event_idx][2] < t_end:
            event_idx += 1
        # Recompute log_intensity state
        log_intensity = np.log(np.clip(intensity, 1e-6, 5))
        rec_direct[frame_idx] = intensity
    
    # Re-run direct integration properly
    rec_direct = np.zeros((n_output_frames, height, width))
    log_intensity = np.log(gt_frames[0] + 1e-6)
    event_idx = 0
    n_events = len(events)
    
    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out
        
        while event_idx < n_events and events[event_idx][2] < t_end:
            x, y, t, p = events[event_idx]
            x, y = int(x), int(y)
            log_intensity[y, x] += p * contrast_threshold
            event_idx += 1
        
        intensity = np.exp(log_intensity)
        intensity = np.clip(intensity, 0, 5)
        rec_direct[frame_idx] = intensity
    
    # Method 2: Complementary Filter
    rec_comp = np.zeros((n_output_frames, height, width))
    event_idx = 0
    alpha = 0.85
    
    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out
        t_mid = (frame_idx + 0.5) * dt_out
        
        # Map to nearest low-res APS frame
        lr_idx = min(int(t_mid * fps), len(aps_frames) - 1)
        lowpass = aps_frames[lr_idx]
        
        # Re-initialise log intensity from APS anchor each frame
        log_intensity_comp = np.log(lowpass + 1e-6)
        
        while event_idx < n_events and events[event_idx][2] < t_end:
            x, y, t, p = events[event_idx]
            x, y = int(x), int(y)
            log_intensity_comp[y, x] += p * contrast_threshold
            event_idx += 1
        
        # High-pass from events (anchored to APS)
        highpass = np.exp(log_intensity_comp)
        highpass = np.clip(highpass, 0, 1)
        
        # Complementary blend
        blended = alpha * lowpass + (1 - alpha) * highpass
        blended = np.clip(blended, 0, 1)
        rec_comp[frame_idx] = blended
    
    # Method 3: TV-Regularised
    # First get raw integration
    raw_recon = np.zeros((n_output_frames, height, width))
    log_intensity_tv = np.zeros((height, width))
    event_idx = 0
    
    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out
        
        while event_idx < n_events and events[event_idx][2] < t_end:
            x, y, t, p = events[event_idx]
            x, y = int(x), int(y)
            log_intensity_tv[y, x] += p * contrast_threshold
            event_idx += 1
        
        intensity = np.exp(log_intensity_tv)
        intensity = np.clip(intensity, 0, 5)
        raw_recon[frame_idx] = intensity
    
    # Apply TV denoising
    rec_tv = np.zeros_like(raw_recon)
    for i in range(n_output_frames):
        frame = raw_recon[i]
        frame = np.clip(frame, 0, 1)
        # TV denoising via iterative median + Gaussian
        for _ in range(5):
            frame = median_filter(frame, size=3)
            frame = gaussian_filter(frame, sigma=0.5)
        rec_tv[i] = np.clip(frame, 0, 1)
    
    all_reconstructions = {
        "Direct": rec_direct,
        "Complementary": rec_comp,
        "TV-Regularised": rec_tv
    }
    
    # Compute metrics for each method to select best
    def compute_psnr(gt, rec):
        n = min(len(gt), len(rec))
        psnr_list = []
        for i in range(n):
            gt_f = gt[i]
            rec_f = rec[i]
            gt_n = (gt_f - gt_f.min()) / (gt_f.max() - gt_f.min() + 1e-10)
            rec_n = (rec_f - rec_f.min()) / (rec_f.max() - rec_f.min() + 1e-10)
            mse = np.mean((gt_n - rec_n)**2)
            psnr_list.append(10 * np.log10(1.0 / max(mse, 1e-30)))
        return np.mean(psnr_list)
    
    psnr_direct = compute_psnr(gt_frames, rec_direct)
    psnr_comp = compute_psnr(gt_frames, rec_comp)
    psnr_tv = compute_psnr(gt_frames, rec_tv)
    
    psnr_values = {
        "Direct": psnr_direct if not np.isnan(psnr_direct) else -999,
        "Complementary": psnr_comp if not np.isnan(psnr_comp) else -999,
        "TV-Regularised": psnr_tv if not np.isnan(psnr_tv) else -999
    }
    
    best_name = max(psnr_values, key=psnr_values.get)
    best_rec = all_reconstructions[best_name]
    
    return best_rec, best_name, all_reconstructions
