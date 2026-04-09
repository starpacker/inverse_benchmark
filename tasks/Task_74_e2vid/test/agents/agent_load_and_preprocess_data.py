import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter, median_filter

def load_and_preprocess_data(height, width, n_frames, contrast_threshold, noise_rate, fps, seed):
    """
    Generate synthetic video sequence and simulate event camera data.
    
    Returns:
        gt_frames: Ground truth intensity frames (n_frames, height, width)
        events: List of events (x, y, t, polarity)
        aps_frames: Simulated APS (low-pass filtered) frames
        t_total: Total time duration
    """
    rng = np.random.default_rng(seed)
    
    # Generate synthetic video: rotating + translating geometric objects
    frames = np.zeros((n_frames, height, width))
    Y, X = np.mgrid[:height, :width]

    for t in range(n_frames):
        img = np.zeros((height, width))

        # Moving circle
        cx = width / 2 + 10 * np.sin(2 * np.pi * t / n_frames)
        cy = height / 2 + 8 * np.cos(2 * np.pi * t / n_frames)
        r = 12
        mask = (X - cx)**2 + (Y - cy)**2 < r**2
        img[mask] = 0.8

        # Moving bar
        angle = np.pi * t / n_frames
        bar_cx = width / 4
        bar_cy = height / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = (X - bar_cx) * cos_a + (Y - bar_cy) * sin_a
        y_rot = -(X - bar_cx) * sin_a + (Y - bar_cy) * cos_a
        bar_mask = (np.abs(x_rot) < 15) & (np.abs(y_rot) < 3)
        img[bar_mask] = 0.6

        # Gradient background
        img += 0.1 * (1 + np.sin(2 * np.pi * X / width + 0.5 * t)
                       + np.cos(2 * np.pi * Y / height)) / 3

        # Blinking point
        img[10, 3 * width // 4] = 0.5 * (1 + np.sin(4 * np.pi * t / n_frames))

        img = np.clip(img, 0.01, 1.0)
        frames[t] = img

    gt_frames = frames
    
    # Generate events from frame differences
    log_frames = np.log(gt_frames + 1e-6)
    events = []
    ref_log = log_frames[0].copy()

    dt = 1.0 / fps
    for t_idx in range(1, n_frames):
        t = t_idx * dt
        diff = log_frames[t_idx] - ref_log

        # Positive events
        pos_ys, pos_xs = np.where(diff >= contrast_threshold)
        for y, x in zip(pos_ys, pos_xs):
            events.append((x, y, t, 1))
        ref_log[pos_ys, pos_xs] = log_frames[t_idx, pos_ys, pos_xs]

        # Negative events
        neg_ys, neg_xs = np.where(diff <= -contrast_threshold)
        for y, x in zip(neg_ys, neg_xs):
            events.append((x, y, t, -1))
        ref_log[neg_ys, neg_xs] = log_frames[t_idx, neg_ys, neg_xs]

        # Add noise events
        n_noise = int(noise_rate * height * width)
        for _ in range(n_noise):
            nx = rng.integers(0, width)
            ny = rng.integers(0, height)
            np_pol = rng.choice([-1, 1])
            nt = t + rng.random() * dt
            events.append((nx, ny, nt, np_pol))

    # Sort by time
    events.sort(key=lambda e: e[2])
    
    # Create low-res APS frames (blurred GT as proxy)
    aps_frames = np.array([gaussian_filter(f, sigma=1.5) for f in gt_frames])
    
    t_total = n_frames / fps
    
    return gt_frames, events, aps_frames, t_total
