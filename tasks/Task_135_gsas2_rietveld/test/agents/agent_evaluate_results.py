import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def evaluate_results(data, inversion_result, results_dir):
    """
    Evaluate refinement results, compute metrics, and generate visualizations.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data
    inversion_result : dict
        Dictionary containing inversion results
    results_dir : str
        Directory to save results
    
    Returns:
    --------
    dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    tt = data['tt']
    y_obs = data['y_obs']
    y_true = data['y_true']
    weights = data['weights']
    true_params = data['true_params']
    initial_params = data['initial_params']
    wavelength = data['wavelength']
    refs = data['refs']
    tt_min = data['tt_min']
    tt_max = data['tt_max']
    npts = data['npts']
    
    refined_params = inversion_result['refined_params']
    y_calc = inversion_result['y_calc']
    
    # Compute Rwp
    def rwp_val(yobs, ycalc, w):
        return np.sqrt(np.sum(w*w*(yobs-ycalc)**2) / np.sum(w*w*yobs**2)) * 100
    
    rwp = rwp_val(y_obs, y_calc, weights)
    
    # Compute other metrics
    np_ = len(refined_params)
    rexp = np.sqrt((npts - np_) / np.sum(weights*weights*y_obs**2)) * 100
    gof = rwp / rexp
    chi2 = np.sum(weights*weights*(y_obs-y_calc)**2) / (npts - np_)
    
    lat_re = abs(refined_params[0]-true_params[0])/true_params[0]*100
    cc = float(np.corrcoef(y_obs, y_calc)[0, 1])
    
    peak_max = np.max(y_true)
    mse = np.mean((y_true - y_calc)**2)
    psnr = 10*np.log10(peak_max**2/mse) if mse > 0 else 999.0
    
    scale_re = abs(refined_params[1]-true_params[1])/true_params[1]*100
    U_re = abs(refined_params[2]-true_params[2])/abs(true_params[2])*100
    W_re = abs(refined_params[4]-true_params[4])/abs(true_params[4])*100
    eta_re = abs(refined_params[5]-true_params[5])/abs(true_params[5])*100
    
    # Print results
    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    for name, true, ref, re_v in [
        ('a (Å)', true_params[0], refined_params[0], lat_re),
        ('scale', true_params[1], refined_params[1], scale_re),
        ('U', true_params[2], refined_params[2], U_re),
        ('W', true_params[4], refined_params[4], W_re),
        ('η', true_params[5], refined_params[5], eta_re),
    ]:
        print(f"  {name:8s}: {ref:10.5f}  (true {true:.4f}, RE {re_v:.3f}%)")
    print(f"  V       : {refined_params[3]:10.5f}  (true {true_params[3]:.4f})")
    print(f"  bg: {[f'{b:.2f}' for b in refined_params[6:]]}")
    print(f"  true bg: {list(true_params[6:])}")
    print(f"\n  Rwp={rwp:.3f}%  Rexp={rexp:.3f}%  GoF={gof:.3f}  χ²={chi2:.4f}")
    print(f"  CC={cc:.6f}  PSNR={psnr:.2f} dB  lat_RE={lat_re:.4f}%")
    
    # Count peaks in range
    n_peaks = 0
    for q2, _, _ in refs:
        d = true_params[0] / np.sqrt(q2)
        ss = wavelength / (2*d)
        if abs(ss) < 1:
            tp = 2*np.degrees(np.arcsin(ss))
            if tt_min <= tp <= tt_max:
                n_peaks += 1
    
    # Build metrics dictionary
    metrics = {
        "task_name": "gsas2_rietveld",
        "method": "Pure-Python Rietveld (least-squares, pseudo-Voigt, Caglioti FWHM)",
        "crystal_system": "CeO2, Fm-3m, fluorite",
        "wavelength_angstrom": wavelength,
        "true_lattice_a": float(true_params[0]),
        "refined_lattice_a": round(float(refined_params[0]), 5),
        "lattice_param_RE_percent": round(lat_re, 4),
        "Rwp_percent": round(rwp, 3),
        "Rexp_percent": round(rexp, 3),
        "GoF": round(gof, 3),
        "chi_squared_reduced": round(chi2, 4),
        "pattern_CC": round(cc, 6),
        "PSNR_dB": round(psnr, 2),
        "scale_true": float(true_params[1]),
        "scale_refined": round(float(refined_params[1]), 4),
        "scale_RE_percent": round(scale_re, 2),
        "U": round(float(refined_params[2]), 5),
        "V": round(float(refined_params[3]), 5),
        "W": round(float(refined_params[4]), 5),
        "eta": round(float(refined_params[5]), 4),
        "num_peaks": n_peaks,
        "two_theta_range_deg": f"{tt_min}-{tt_max}",
        "num_points": npts,
        "converged": inversion_result['converged'],
        "n_evals": inversion_result['n_evals']
    }
    
    # Save metrics
    mp = os.path.join(results_dir, 'metrics.json')
    with open(mp, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {mp}")
    
    # Save data arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'),
            {'two_theta': tt, 'y_true': y_true, 'y_obs': y_obs,
             'true_params': true_params}, allow_pickle=True)
    np.save(os.path.join(results_dir, 'reconstruction.npy'),
            {'two_theta': tt, 'y_calc': y_calc,
             'refined_params': refined_params}, allow_pickle=True)
    print("Saved ground_truth.npy, reconstruction.npy")
    
    # Generate visualization
    print("Generating visualization...")
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.08, 1.3], hspace=0.32)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(tt, y_obs, 'b-', lw=0.3, alpha=0.5, label='Observed')
    ax1.plot(tt, y_calc, 'r-', lw=0.8, alpha=0.85, label='Calculated (Rietveld)')
    for q2, _, _ in refs:
        d = refined_params[0]/np.sqrt(q2)
        s = wavelength/(2*d)
        if abs(s) < 1:
            tp = 2*np.degrees(np.arcsin(s))
            if tt_min <= tp <= tt_max:
                ax1.axvline(tp, ymin=0, ymax=0.02, color='green', lw=0.6, alpha=0.5)
    ax1.set_ylabel('Intensity (counts)', fontsize=12)
    ax1.set_title(f'Rietveld Fit: CeO2 (Fm-3m, Cu Kα)  |  Rwp={rwp:.2f}%  '
                  f'CC={cc:.5f}  GoF={gof:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(tt_min, tt_max)
    ax1.grid(True, alpha=0.2)
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    diff = y_obs - y_calc
    ax2.plot(tt, diff, 'g-', lw=0.4)
    ax2.axhline(0, color='k', lw=0.5, ls='--')
    ax2.fill_between(tt, diff, 0, alpha=0.1, color='green')
    ax2.set_ylabel('Δ', fontsize=11)
    ax2.set_xlabel('2θ (degrees)', fontsize=12)
    ax2.set_title('Difference (Obs − Calc)', fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    ax3 = fig.add_subplot(gs[3])
    ax3.axis('off')
    rows = [
        ['Param', 'True', 'Initial', 'Refined', 'RE(%)'],
        ['a (Å)', f'{true_params[0]:.4f}', f'{initial_params[0]:.4f}', f'{refined_params[0]:.5f}', f'{lat_re:.4f}'],
        ['Scale', f'{true_params[1]:.2f}', f'{initial_params[1]:.2f}', f'{refined_params[1]:.4f}', f'{scale_re:.2f}'],
        ['U', f'{true_params[2]:.4f}', f'{initial_params[2]:.4f}', f'{refined_params[2]:.5f}', f'{U_re:.2f}'],
        ['W', f'{true_params[4]:.4f}', f'{initial_params[4]:.4f}', f'{refined_params[4]:.5f}', f'{W_re:.2f}'],
        ['η', f'{true_params[5]:.2f}', f'{initial_params[5]:.2f}', f'{refined_params[5]:.4f}', f'{eta_re:.2f}'],
    ]
    tbl = ax3.table(cellText=rows, loc='center', cellLoc='center',
                    colWidths=[0.12, 0.14, 0.14, 0.16, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)
    for j in range(5):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        try:
            v = float(rows[i][4])
            tbl[i, 4].set_facecolor('#C6EFCE' if v < 1 else '#FFEB9C' if v < 5 else '#FFC7CE')
        except ValueError:
            pass
    ax3.set_title('Refined Parameters', fontsize=11, pad=8)
    
    fig.text(0.5, 0.005,
             f'Rwp={rwp:.2f}%  |  GoF={gof:.2f}  |  χ²={chi2:.2f}  |  '
             f'a_RE={lat_re:.4f}%  |  CC={cc:.5f}  |  PSNR={psnr:.1f} dB',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    vp = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(vp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {vp}")
    
    return metrics
