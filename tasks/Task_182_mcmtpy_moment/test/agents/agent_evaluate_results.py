import os

import json

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def source_time_function(t, t0, half_width=0.2):
    """Gaussian source-time function centred at t0."""
    return np.exp(-((t - t0) ** 2) / (2.0 * half_width ** 2))

def radiation_P(strike_deg, dip_deg, rake_deg, azimuth_deg, takeoff_deg):
    """
    P-wave radiation pattern for a double-couple source.
    Aki & Richards (2002), Eq. 4.29.
    """
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    r = np.radians(rake_deg)
    az = np.radians(azimuth_deg)
    ih = np.radians(takeoff_deg)
    phi = az - s

    R = (np.cos(r) * np.sin(d) * np.sin(ih)**2 * np.sin(2 * phi)
         - np.cos(r) * np.cos(d) * np.sin(2 * ih) * np.cos(phi)
         + np.sin(r) * np.sin(2 * d) * (np.cos(ih)**2 - np.sin(ih)**2 * np.sin(phi)**2)
         + np.sin(r) * np.cos(2 * d) * np.sin(2 * ih) * np.sin(phi))
    return R

def forward_operator(params, config, T=None, WIN_INDICES=None, windowed=False):
    """
    Compute synthetic P-wave waveforms at all stations.
    
    Parameters
    ----------
    params : tuple or array
        (strike, dip, rake, log_M0) source parameters
    config : dict
        Configuration parameters
    T : array, optional
        Time array (required if windowed=False)
    WIN_INDICES : list, optional
        Window indices (required if windowed=True)
    windowed : bool
        If True, compute only within signal windows
        
    Returns
    -------
    waveforms : ndarray or list
        Synthetic waveforms. If windowed=False, returns (N_STATIONS x NT) array.
        If windowed=True, returns list of windowed arrays.
    """
    strike, dip, rake, log_M0 = params
    M0 = 10.0 ** log_M0
    
    VP = config['VP']
    N_STATIONS = config['N_STATIONS']
    AZIMUTHS = config['AZIMUTHS']
    DISTANCES = config['DISTANCES']
    TAKEOFFS = config['TAKEOFFS']
    STF_WIDTH = config['STF_WIDTH']
    
    if windowed:
        result = []
        for i in range(N_STATIONS):
            R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
            travel_time = DISTANCES[i] / VP
            amp = R * M0 / DISTANCES[i]
            i0, i1 = WIN_INDICES[i]
            t_win = T[i0:i1]
            stf = source_time_function(t_win, travel_time, half_width=STF_WIDTH)
            result.append(amp * stf)
        return result
    else:
        NT = len(T)
        waveforms = np.zeros((N_STATIONS, NT))
        for i in range(N_STATIONS):
            R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
            travel_time = DISTANCES[i] / VP
            amp = R * M0 / DISTANCES[i]
            stf = source_time_function(T, travel_time, half_width=STF_WIDTH)
            waveforms[i] = amp * stf
        return waveforms

def evaluate_results(data, inversion_result, output_dir="results"):
    """
    Evaluate inversion results and generate visualizations.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data
    inversion_result : dict
        Results from run_inversion
    output_dir : str
        Output directory for results
        
    Returns
    -------
    metrics : dict
        Evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    d_obs = data['d_obs']
    d_clean = data['d_clean']
    WIN_INDICES = data['WIN_INDICES']
    T = data['T']
    config = data['config']
    gt = data['ground_truth']
    
    N_STATIONS = config['N_STATIONS']
    AZIMUTHS = config['AZIMUTHS']
    
    map_est = inversion_result['map_estimate']
    flat = inversion_result['flat_samples']
    chain = inversion_result['chain']
    
    strike_est, dip_est, rake_est, logM0_est = map_est
    M0_est = 10.0 ** logM0_est
    
    GT_STRIKE = gt['strike']
    GT_DIP = gt['dip']
    GT_RAKE = gt['rake']
    GT_LOG_M0 = gt['log_m0']
    
    # Compute reconstruction
    d_recon = forward_operator(
        (strike_est, dip_est, rake_est, logM0_est),
        config,
        T=T,
        windowed=False
    )
    
    def angular_error(est, true, period):
        diff = abs(est - true) % period
        return min(diff, period - diff) / period
    
    def cc_windowed(obs_full, syn_full, i0, i1):
        a = obs_full[i0:i1].copy()
        b = syn_full[i0:i1].copy()
        a -= a.mean()
        b -= b.mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom < 1e-30:
            return 1.0
        return float(np.sum(a * b) / denom)
    
    def psnr_windowed(obs_full, syn_full, i0, i1):
        r = obs_full[i0:i1]
        c = syn_full[i0:i1]
        mse = np.mean((r - c)**2)
        peak = np.max(np.abs(r))
        if mse < 1e-30 or peak < 1e-30:
            return 100.0
        return 20.0 * np.log10(peak / np.sqrt(mse))
    
    ccs = []
    psnrs = []
    weights = []
    for i in range(N_STATIONS):
        i0, i1 = WIN_INDICES[i]
        c = cc_windowed(d_obs[i], d_recon[i], i0, i1)
        p = psnr_windowed(d_obs[i], d_recon[i], i0, i1)
        w = np.max(np.abs(d_clean[i]))
        ccs.append(c)
        psnrs.append(p)
        weights.append(w)
        print(f"  Station {i} (az={AZIMUTHS[i]:.0f}deg): CC={c:.4f}, PSNR={p:.1f} dB, amp={w:.2e}")
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    waveform_cc = float(np.sum(np.array(ccs) * weights))
    waveform_psnr = float(np.sum(np.array(psnrs) * weights))
    
    strike_RE = angular_error(strike_est, GT_STRIKE, 360)
    dip_RE = angular_error(dip_est, GT_DIP, 90)
    rake_RE = angular_error(rake_est, GT_RAKE, 360)
    M0_RE = abs(M0_est - 10**GT_LOG_M0) / 10**GT_LOG_M0
    
    metrics = {
        "strike_gt": GT_STRIKE,
        "strike_est": round(float(strike_est), 2),
        "strike_RE": round(float(strike_RE), 5),
        "dip_gt": GT_DIP,
        "dip_est": round(float(dip_est), 2),
        "dip_RE": round(float(dip_RE), 5),
        "rake_gt": GT_RAKE,
        "rake_est": round(float(rake_est), 2),
        "rake_RE": round(float(rake_RE), 5),
        "M0_gt": float(10 ** GT_LOG_M0),
        "M0_est": round(float(M0_est), 2),
        "M0_RE": round(float(M0_RE), 5),
        "waveform_CC": round(waveform_cc, 5),
        "waveform_PSNR_dB": round(waveform_psnr, 2),
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n[METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save arrays
    np.save(os.path.join(output_dir, "ground_truth.npy"), d_obs)
    np.save(os.path.join(output_dir, "reconstruction.npy"), d_recon)
    
    # Visualization
    NWALKERS = chain.shape[1]
    NDIM = 4
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Task 182 — Seismic Moment Tensor Inversion (MCMC)",
                 fontsize=16, fontweight="bold", y=0.98)
    
    # (a) Parameter comparison table
    ax_a = fig.add_axes([0.05, 0.55, 0.42, 0.38])
    ax_a.axis("off")
    ax_a.set_title("(a) Source Parameter Comparison", fontsize=13, fontweight="bold", pad=10)
    
    table_data = [
        ["Parameter", "Ground Truth", "MAP Estimate", "Rel. Error"],
        ["Strike (deg)", f"{GT_STRIKE:.1f}", f"{strike_est:.2f}", f"{strike_RE:.4f}"],
        ["Dip (deg)", f"{GT_DIP:.1f}", f"{dip_est:.2f}", f"{dip_RE:.4f}"],
        ["Rake (deg)", f"{GT_RAKE:.1f}", f"{rake_est:.2f}", f"{rake_RE:.4f}"],
        ["log10(M0)", f"{GT_LOG_M0:.2f}", f"{logM0_est:.4f}", f"{M0_RE:.4f}"],
        ["Waveform CC", "", f"{waveform_cc:.4f}", ""],
        ["Waveform PSNR", "", f"{waveform_psnr:.1f} dB", ""],
    ]
    colours = [["#d0d0d0"] * 4] + [["#ffffff"] * 4] * 6
    table = ax_a.table(cellText=table_data, cellColours=colours,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
    
    # (b) Waveform fits
    ax_b = fig.add_axes([0.55, 0.55, 0.40, 0.38])
    ax_b.set_title("(b) Waveform Fits (selected stations)", fontsize=13, fontweight="bold")
    sel = [0, 2, 4, 6]
    for j, idx in enumerate(sel):
        i0, i1 = WIN_INDICES[idx]
        t_win = T[i0:i1]
        scale = max(np.max(np.abs(d_obs[idx, i0:i1])), 1e-30)
        offset = j * 2.5
        ax_b.plot(t_win, d_obs[idx, i0:i1] / scale + offset, "k", lw=1.2,
                  label="Obs" if j == 0 else "")
        ax_b.plot(t_win, d_recon[idx, i0:i1] / scale + offset, "r--", lw=1.2,
                  label="Syn" if j == 0 else "")
        ax_b.text(t_win[-1] + 0.2, offset, f"Sta {idx+1}\naz={AZIMUTHS[idx]:.0f}deg\nCC={ccs[idx]:.3f}",
                  fontsize=8, va="center")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Normalised amplitude + offset")
    ax_b.legend(loc="upper left", fontsize=9)
    
    # (c) Posterior distributions
    labels_post = ["Strike (deg)", "Dip (deg)", "Rake (deg)", "log10(M0)"]
    truths_post = [GT_STRIKE, GT_DIP, GT_RAKE, GT_LOG_M0]
    for k in range(NDIM):
        ax_sub = fig.add_axes([0.07 + k * 0.22, 0.28, 0.18, 0.18])
        ax_sub.hist(flat[:, k], bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="none")
        ax_sub.axvline(truths_post[k], color="red", lw=2, ls="--", label="GT")
        ax_sub.axvline(map_est[k], color="green", lw=2, ls="-", label="MAP")
        ax_sub.set_xlabel(labels_post[k], fontsize=10)
        ax_sub.set_ylabel("Density" if k == 0 else "", fontsize=9)
        if k == 0:
            ax_sub.legend(fontsize=8)
            ax_sub.set_title("(c) Posterior Distributions", fontsize=13, fontweight="bold",
                             loc="left", pad=12)
        ax_sub.tick_params(labelsize=8)
    
    # (d) MCMC traces
    param_names = ["Strike", "Dip", "Rake", "log10(M0)"]
    for k in range(NDIM):
        ax_trace = fig.add_axes([0.07, 0.02 + k * 0.06, 0.88, 0.05])
        for w in range(0, NWALKERS, 4):
            ax_trace.plot(chain[:, w, k], alpha=0.25, lw=0.3, color="C0")
        ax_trace.axhline(truths_post[k], color="red", lw=1.2, ls="--")
        ax_trace.set_ylabel(param_names[k], fontsize=8)
        ax_trace.tick_params(labelsize=7)
        if k == 0:
            ax_trace.set_xlabel("MCMC Step", fontsize=9)
        else:
            ax_trace.set_xticklabels([])
        if k == NDIM - 1:
            ax_trace.set_title("(d) MCMC Parameter Traces", fontsize=13,
                               fontweight="bold", loc="left", pad=8)
    
    plt.savefig(os.path.join(output_dir, "reconstruction_result.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Saved {output_dir}/reconstruction_result.png")
    
    return metrics
