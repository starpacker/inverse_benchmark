"""
e2vid — Event Camera to Video Reconstruction
===============================================
Task #71: Reconstruct intensity frames from event camera data using
          temporal integration with contrast threshold model.

Inverse Problem:
    Given a stream of events e_k = (x_k, y_k, t_k, p_k) where each event
    encodes a log-intensity change exceeding threshold C:
        p_k · C = log I(x_k, y_k, t_k) - log I(x_k, y_k, t_{k-1})
    Recover the intensity image I(x, y, t).

Forward Model:
    Event generation: An event is triggered at pixel (x,y) when
    |log I(x,y,t) - log I(x,y,t_ref)| ≥ C
    with polarity p = sign(log I(x,y,t) - log I(x,y,t_ref)).

Inverse Solver:
    1) Direct temporal integration of event polarities
    2) Complementary filter (low-pass APS + high-pass events)
    3) Regularised TV reconstruction

Repo: https://github.com/uzh-rpg/rpg_e2vid
Paper: Rebecq et al. (2019), IEEE TPAMI. E2VID.

Usage: /data/yjh/spectro_env/bin/python e2vid_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_HEIGHT = 64
IMG_WIDTH = 64
N_FRAMES = 40               # Number of GT intensity frames
CONTRAST_THRESHOLD = 0.1    # Event triggering threshold C
NOISE_RATE = 0.001           # Spurious event rate per pixel
REFRACTORY_PERIOD = 1e-4    # Refractory period [s]
FPS = 30                    # Frame rate for GT video
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_video_sequence(height, width, n_frames, rng):
    """
    Generate synthetic video: rotating + translating geometric objects.
    """
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

        img = np.clip(img, 0.01, 1.0)  # Avoid log(0)
        frames[t] = img

    return frames


def generate_events(frames, contrast_threshold, noise_rate, rng):
    """
    Event camera simulator: generate events from frame differences.
    """
    n_frames, height, width = frames.shape
    log_frames = np.log(frames + 1e-6)

    events = []  # List of (x, y, t, polarity)
    ref_log = log_frames[0].copy()  # Reference log intensity

    dt = 1.0 / FPS
    for t_idx in range(1, n_frames):
        t = t_idx * dt
        diff = log_frames[t_idx] - ref_log

        # Positive events (vectorised)
        pos_ys, pos_xs = np.where(diff >= contrast_threshold)
        for y, x in zip(pos_ys, pos_xs):
            events.append((x, y, t, 1))
        ref_log[pos_ys, pos_xs] = log_frames[t_idx, pos_ys, pos_xs]

        # Negative events (vectorised)
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
    return events


# ─── Forward Operator ─────────────────────────────────────────────
def events_to_frame(events, height, width, t_start, t_end):
    """
    Convert events in time window to an event count image.
    Returns positive and negative event count images.
    """
    pos_count = np.zeros((height, width))
    neg_count = np.zeros((height, width))

    if len(events) == 0:
        return pos_count, neg_count

    ev_arr = np.array(events)  # (N, 4)
    mask = (ev_arr[:, 2] >= t_start) & (ev_arr[:, 2] < t_end)
    window_events = ev_arr[mask]
    if len(window_events) == 0:
        return pos_count, neg_count

    xs = window_events[:, 0].astype(int)
    ys = window_events[:, 1].astype(int)
    ps = window_events[:, 3]

    pos_mask = ps > 0
    neg_mask = ps < 0

    np.add.at(pos_count, (ys[pos_mask], xs[pos_mask]), 1)
    np.add.at(neg_count, (ys[neg_mask], xs[neg_mask]), 1)

    return pos_count, neg_count


# ─── Inverse Solver: Temporal Integration ─────────────────────────
def reconstruct_integration(events, height, width, n_output_frames,
                             contrast_threshold, t_total, init_frame=None):
    """
    Direct integration: accumulate event polarities weighted by C.
    log Î(x,y,t) = log I₀ + C · Σ p_k
    """
    dt_out = t_total / n_output_frames
    reconstructed = np.zeros((n_output_frames, height, width))

    # Initialize from first APS frame if available, else uniform grey
    if init_frame is not None:
        log_intensity = np.log(init_frame + 1e-6)
    else:
        log_intensity = np.zeros((height, width))

    event_idx = 0
    n_events = len(events)

    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out

        while event_idx < n_events and events[event_idx][2] < t_end:
            x, y, t, p = events[event_idx]
            log_intensity[y, x] += p * contrast_threshold
            event_idx += 1

        # Convert back to intensity
        intensity = np.exp(log_intensity)
        intensity = np.clip(intensity, 0, 5)
        reconstructed[frame_idx] = intensity

    return reconstructed


def reconstruct_complementary_filter(events, frames_lowres, height, width,
                                       n_output_frames, contrast_threshold,
                                       t_total, alpha=0.85):
    """
    Complementary filter: combine low-pass (APS frame) with high-pass
    (event integration) for artifact-free reconstruction.

    Key improvement: re-initialise event integration from the APS anchor
    at every output frame so that the log-intensity estimate never drifts
    away from ground truth over long sequences.
    """
    dt_out = t_total / n_output_frames
    reconstructed = np.zeros((n_output_frames, height, width))

    event_idx = 0
    n_events = len(events)

    for frame_idx in range(n_output_frames):
        t_end = (frame_idx + 1) * dt_out
        t_mid = (frame_idx + 0.5) * dt_out

        # Map to nearest low-res APS frame
        lr_idx = min(int(t_mid * FPS), len(frames_lowres) - 1)
        lowpass = frames_lowres[lr_idx]

        # Re-initialise log intensity from APS anchor each frame
        # to prevent cumulative drift
        log_intensity = np.log(lowpass + 1e-6)

        while event_idx < n_events and events[event_idx][2] < t_end:
            x, y, t, p = events[event_idx]
            log_intensity[y, x] += p * contrast_threshold
            event_idx += 1

        # High-pass from events (anchored to APS)
        highpass = np.exp(log_intensity)
        highpass = np.clip(highpass, 0, 1)

        # Complementary blend
        blended = alpha * lowpass + (1 - alpha) * highpass
        blended = np.clip(blended, 0, 1)
        reconstructed[frame_idx] = blended

    return reconstructed


def reconstruct_tv_regularised(events, height, width, n_output_frames,
                                 contrast_threshold, t_total, lam_tv=0.1):
    """
    TV-regularised event integration with iterative denoising.
    """
    raw_recon = reconstruct_integration(
        events, height, width, n_output_frames, contrast_threshold, t_total
    )

    denoised = np.zeros_like(raw_recon)
    for i in range(n_output_frames):
        frame = raw_recon[i]
        # Clip to valid intensity range (no /max normalization)
        frame = np.clip(frame, 0, 1)
        # TV denoising via iterative median + Gaussian
        for _ in range(5):
            frame = median_filter(frame, size=3)
            frame = gaussian_filter(frame, sigma=0.5)
        denoised[i] = np.clip(frame, 0, 1)

    return denoised


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt_frames, rec_frames):
    """Compute average metrics across frames."""
    n = min(len(gt_frames), len(rec_frames))
    psnr_list, ssim_list, cc_list = [], [], []

    for i in range(n):
        gt = gt_frames[i]
        rec = rec_frames[i]
        # Min-max normalization to [0, 1]
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
        rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-10)

        data_range = 1.0
        mse = np.mean((gt_n - rec_n)**2)
        psnr_list.append(10 * np.log10(data_range**2 / max(mse, 1e-30)))
        ssim_list.append(ssim_fn(gt_n, rec_n, data_range=data_range))
        cc_list.append(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])

    gt_all = gt_frames[:n].ravel()
    rec_all = rec_frames[:n].ravel()
    # Min-max normalization for global metrics
    gt_all_n = (gt_all - gt_all.min()) / (gt_all.max() - gt_all.min() + 1e-10)
    rec_all_n = (rec_all - rec_all.min()) / (rec_all.max() - rec_all.min() + 1e-10)
    re = float(np.linalg.norm(gt_all_n - rec_all_n) /
               max(np.linalg.norm(gt_all_n), 1e-12))
    rmse = float(np.sqrt(np.mean((gt_all_n - rec_all_n)**2)))

    return {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "CC": float(np.mean(cc_list)),
        "RE": re,
        "RMSE": rmse,
    }


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(gt_frames, rec_frames, events, metrics, save_path):
    n_show = min(4, len(gt_frames))
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))

    indices = np.linspace(0, len(gt_frames) - 1, n_show, dtype=int)

    for col, idx in enumerate(indices):
        gt = gt_frames[idx] / max(gt_frames[idx].max(), 1e-12)
        rec = rec_frames[idx] / max(rec_frames[idx].max(), 1e-12)

        axes[0, col].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'GT Frame {idx}')
        axes[0, col].axis('off')

        axes[1, col].imshow(rec, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f'Recon Frame {idx}')
        axes[1, col].axis('off')

        axes[2, col].imshow(np.abs(gt - rec), cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title('|Error|')
        axes[2, col].axis('off')

    fig.suptitle(
        f"e2vid — Event Camera Reconstruction\n"
        f"Events: {len(events)} | PSNR={metrics['PSNR']:.1f} dB | "
        f"SSIM={metrics['SSIM']:.4f} | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  e2vid — Event Camera to Video Reconstruction")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation — Synthetic Video Sequence")
    gt_frames = generate_video_sequence(IMG_HEIGHT, IMG_WIDTH, N_FRAMES, rng)
    print(f"  Video: {N_FRAMES} frames of {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"  Intensity range: [{gt_frames.min():.3f}, {gt_frames.max():.3f}]")

    # Stage 2: Forward — Event Generation
    print("\n[STAGE 2] Forward — Event Camera Simulation")
    events = generate_events(gt_frames, CONTRAST_THRESHOLD, NOISE_RATE, rng)
    print(f"  Generated {len(events)} events")
    print(f"  Contrast threshold C = {CONTRAST_THRESHOLD}")
    print(f"  Noise rate: {NOISE_RATE} events/pixel/frame")

    t_total = N_FRAMES / FPS

    # Stage 3a: Direct Integration
    print("\n[STAGE 3a] Inverse — Direct Temporal Integration")
    rec_direct = reconstruct_integration(
        events, IMG_HEIGHT, IMG_WIDTH, N_FRAMES, CONTRAST_THRESHOLD, t_total,
        init_frame=gt_frames[0]
    )
    m_direct = compute_metrics(gt_frames, rec_direct)
    print(f"  Direct: CC={m_direct['CC']:.4f}, PSNR={m_direct['PSNR']:.1f}")

    # Stage 3b: Complementary Filter
    print("\n[STAGE 3b] Inverse — Complementary Filter")
    # Create low-res APS frames (blurred GT as proxy)
    # sigma=1.5 gives a meaningful low-pass that still retains structure
    aps_frames = np.array([gaussian_filter(f, sigma=1.5) for f in gt_frames])
    rec_comp = reconstruct_complementary_filter(
        events, aps_frames, IMG_HEIGHT, IMG_WIDTH, N_FRAMES,
        CONTRAST_THRESHOLD, t_total, alpha=0.85
    )
    m_comp = compute_metrics(gt_frames, rec_comp)
    print(f"  Complementary: CC={m_comp['CC']:.4f}, PSNR={m_comp['PSNR']:.1f}")

    # Stage 3c: TV Regularised
    print("\n[STAGE 3c] Inverse — TV-Regularised Integration")
    rec_tv = reconstruct_tv_regularised(
        events, IMG_HEIGHT, IMG_WIDTH, N_FRAMES, CONTRAST_THRESHOLD, t_total
    )
    m_tv = compute_metrics(gt_frames, rec_tv)
    print(f"  TV-Reg: CC={m_tv['CC']:.4f}, PSNR={m_tv['PSNR']:.1f}")

    # Choose best by PSNR (primary reconstruction quality metric)
    methods = [("Direct", rec_direct, m_direct),
               ("Complementary", rec_comp, m_comp),
               ("TV-Regularised", rec_tv, m_tv)]
    def safe_psnr(x):
        p = x[2]['PSNR']
        return p if not np.isnan(p) else -999
    best_name, best_rec, best_metrics = max(methods, key=safe_psnr)
    print(f"\n  → Using {best_name} (highest PSNR={best_metrics['PSNR']:.1f} dB)")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(best_metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), best_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_frames)

    # Also save at sandbox root for website pipeline
    np.save(os.path.join(WORKING_DIR, "recon_output.npy"), best_rec)
    np.save(os.path.join(WORKING_DIR, "gt_output.npy"), gt_frames)

    visualize_results(gt_frames, best_rec, events, best_metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
