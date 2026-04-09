import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the evaluate_results function verbatim from Reference B
# ============================================================
def evaluate_results(data, inversion_result, results_dir):
    """
    Evaluate refinement results, compute metrics, and generate visualizations.
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


# ============================================================
# Main test logic
# ============================================================
def main():
    data_paths = ['/data/yjh/gsas2_rietveld_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # ============================================================
    # Determine execution pattern
    # ============================================================
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("\n=== Pattern 2: Chained Execution ===")
        print(f"Running run_inversion with outer data to get operator...")
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion (outer): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print(f"Running operator with inner data...")
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running operator (inner): {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("\n=== Pattern 1: Direct Execution ===")
        print(f"Running run_inversion with outer data...")
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        std_result = std_output
    
    # ============================================================
    # Evaluation Phase
    # ============================================================
    print("\n" + "="*60)
    print("  EVALUATION PHASE")
    print("="*60)
    
    # Extract the input data (first arg should be the data dict)
    input_data = args[0] if len(args) > 0 else kwargs.get('data', None)
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation.")
        sys.exit(1)
    
    # Evaluate agent result
    results_dir_agent = './results_agent'
    print("\n--- Evaluating AGENT result ---")
    try:
        metrics_agent = evaluate_results(input_data, agent_result, results_dir_agent)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    results_dir_std = './results_std'
    print("\n--- Evaluating STANDARD result ---")
    try:
        metrics_std = evaluate_results(input_data, std_result, results_dir_std)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ============================================================
    # Comparison and Reporting
    # ============================================================
    print("\n" + "="*60)
    print("  COMPARISON REPORT")
    print("="*60)
    
    # Extract key metrics for comparison
    # Rwp: lower is better
    # CC: higher is better (closer to 1)
    # PSNR: higher is better
    # GoF: closer to 1 is better (lower is better when > 1)
    # lattice_param_RE: lower is better
    
    rwp_agent = metrics_agent['Rwp_percent']
    rwp_std = metrics_std['Rwp_percent']
    
    cc_agent = metrics_agent['pattern_CC']
    cc_std = metrics_std['pattern_CC']
    
    psnr_agent = metrics_agent['PSNR_dB']
    psnr_std = metrics_std['PSNR_dB']
    
    gof_agent = metrics_agent['GoF']
    gof_std = metrics_std['GoF']
    
    lat_re_agent = metrics_agent['lattice_param_RE_percent']
    lat_re_std = metrics_std['lattice_param_RE_percent']
    
    converged_agent = metrics_agent['converged']
    converged_std = metrics_std['converged']
    
    print(f"\n  {'Metric':<25s} {'Agent':>12s} {'Standard':>12s} {'Status':>10s}")
    print(f"  {'-'*60}")
    
    all_pass = True
    
    # Rwp (lower is better) - allow 10% margin
    rwp_ok = rwp_agent <= rwp_std * 1.10 + 0.5  # 10% relative + 0.5 absolute margin
    status = "PASS" if rwp_ok else "FAIL"
    if not rwp_ok:
        all_pass = False
    print(f"  {'Rwp (%)':<25s} {rwp_agent:>12.3f} {rwp_std:>12.3f} {status:>10s}")
    
    # CC (higher is better) - allow small margin
    cc_ok = cc_agent >= cc_std * 0.999  # very tight for CC near 1
    status = "PASS" if cc_ok else "FAIL"
    if not cc_ok:
        all_pass = False
    print(f"  {'Pattern CC':<25s} {cc_agent:>12.6f} {cc_std:>12.6f} {status:>10s}")
    
    # PSNR (higher is better) - allow 10% margin
    psnr_ok = psnr_agent >= psnr_std * 0.90
    status = "PASS" if psnr_ok else "FAIL"
    if not psnr_ok:
        all_pass = False
    print(f"  {'PSNR (dB)':<25s} {psnr_agent:>12.2f} {psnr_std:>12.2f} {status:>10s}")
    
    # GoF (lower is better when >= 1) - allow 10% margin
    gof_ok = gof_agent <= gof_std * 1.10 + 0.1
    status = "PASS" if gof_ok else "FAIL"
    if not gof_ok:
        all_pass = False
    print(f"  {'GoF':<25s} {gof_agent:>12.3f} {gof_std:>12.3f} {status:>10s}")
    
    # Lattice param RE (lower is better) - allow margin
    lat_ok = lat_re_agent <= lat_re_std * 1.10 + 0.01  # 10% relative + 0.01 absolute
    status = "PASS" if lat_ok else "FAIL"
    if not lat_ok:
        all_pass = False
    print(f"  {'Lattice RE (%)':<25s} {lat_re_agent:>12.4f} {lat_re_std:>12.4f} {status:>10s}")
    
    # Convergence check
    conv_ok = converged_agent or (not converged_std)  # agent should converge if std converges
    status = "PASS" if conv_ok else "FAIL"
    if not conv_ok:
        all_pass = False
    print(f"  {'Converged':<25s} {str(converged_agent):>12s} {str(converged_std):>12s} {status:>10s}")
    
    print(f"\n  {'-'*60}")
    
    if all_pass:
        print("\n  ✅ ALL CHECKS PASSED - Agent performance is acceptable.")
        print(f"\n  Scores -> Agent Rwp: {rwp_agent:.3f}%, Standard Rwp: {rwp_std:.3f}%")
        print(f"  Scores -> Agent PSNR: {psnr_agent:.2f} dB, Standard PSNR: {psnr_std:.2f} dB")
        sys.exit(0)
    else:
        print("\n  ❌ SOME CHECKS FAILED - Agent performance degraded significantly.")
        print(f"\n  Scores -> Agent Rwp: {rwp_agent:.3f}%, Standard Rwp: {rwp_std:.3f}%")
        print(f"  Scores -> Agent PSNR: {psnr_agent:.2f} dB, Standard PSNR: {psnr_std:.2f} dB")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)