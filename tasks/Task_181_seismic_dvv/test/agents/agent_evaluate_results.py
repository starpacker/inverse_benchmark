import json

import os

import numpy as np

def evaluate_results(dvv_true: np.ndarray, dvv_est: np.ndarray,
                     t: np.ndarray, ccf_ref: np.ndarray,
                     ccf_matrix: np.ndarray, days: np.ndarray,
                     results_dir: str) -> dict:
    """
    Compute dv/v estimation quality metrics and save results.
    
    Parameters:
        dvv_true: true dv/v values
        dvv_est: estimated dv/v values
        t: time axis array
        ccf_ref: reference CCF
        ccf_matrix: matrix of perturbed CCFs
        days: array of day indices
        results_dir: directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Mean absolute error
    mae = float(np.mean(np.abs(dvv_est - dvv_true)))
    
    # Relative error (fraction of amplitude range)
    amp_range = np.max(dvv_true) - np.min(dvv_true)
    rel_error = float(mae / amp_range) if amp_range > 0 else float('inf')
    
    # Correlation coefficient
    cc = float(np.corrcoef(dvv_true, dvv_est)[0, 1])
    
    # PSNR (treating dv/v time series as 1D signal)
    mse = float(np.mean((dvv_est - dvv_true) ** 2))
    peak = float(np.max(np.abs(dvv_true)))
    if mse > 0 and peak > 0:
        psnr = float(20.0 * np.log10(peak / np.sqrt(mse)))
    else:
        psnr = float('inf')
    
    metrics = {
        "dvv_mae": mae,
        "dvv_relative_error": rel_error,
        "dvv_correlation_coefficient": cc,
        "dvv_psnr_dB": psnr,
    }
    
    # Print metrics
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Visualisation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Reference CCF
    ax = axes[0, 0]
    ax.plot(t, ccf_ref, 'k', lw=0.8)
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(a) Reference CCF")
    ax.set_xlim(t[0], t[-1])
    
    # (b) Reference vs current (one example day)
    ax = axes[0, 1]
    example_day = min(10, len(days) - 1)
    ax.plot(t, ccf_ref, 'k', lw=0.8, label="Reference")
    ax.plot(t, ccf_matrix[example_day], 'r', lw=0.8, alpha=0.7,
            label=f"Current (day {example_day})")
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("(b) Reference vs Perturbed CCF")
    ax.legend(fontsize=9)
    ax.set_xlim(t[0], t[-1])
    
    # (c) True vs estimated dv/v
    ax = axes[1, 0]
    ax.plot(days, dvv_true * 100, 'k-o', ms=3, lw=1.2, label="True dv/v")
    ax.plot(days, dvv_est * 100, 'r-s', ms=3, lw=1.2, label="Estimated dv/v")
    ax.set_xlabel("Day")
    ax.set_ylabel("dv/v (%)")
    ax.set_title("(c) dv/v Time Series")
    ax.legend(fontsize=9)
    
    # (d) Residual
    ax = axes[1, 1]
    residual = (dvv_true - dvv_est) * 100
    ax.bar(days, residual, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Residual dv/v (%)")
    ax.set_title("(d) Residual (True − Estimated)")
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {save_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), dvv_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), dvv_est)
    print("Arrays saved.")
    
    return metrics
