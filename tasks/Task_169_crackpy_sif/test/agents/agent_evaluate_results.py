import os

import json

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def evaluate_results(data_dict, result):
    """
    Evaluate the inversion results and generate outputs.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing ground truth data and parameters
    result : dict
        Dictionary containing fitted results from inversion
    
    Returns
    -------
    metrics : dict
        Dictionary containing all evaluation metrics
    """
    # Extract data
    K_I_true = data_dict['K_I_true']
    K_II_true = data_dict['K_II_true']
    ux_clean = data_dict['ux_clean']
    uy_clean = data_dict['uy_clean']
    ux_noisy = data_dict['ux_noisy']
    uy_noisy = data_dict['uy_noisy']
    r_flat = data_dict['r_flat']
    theta_flat = data_dict['theta_flat']
    snr_db = data_dict['snr_db']
    N_terms = data_dict['N_terms']
    E = data_dict['E']
    nu = data_dict['nu']
    coeffs_I_true = data_dict['coeffs_I_true']
    coeffs_II_true = data_dict['coeffs_II_true']
    
    K_I_fit = result['K_I_fit']
    K_II_fit = result['K_II_fit']
    ux_fit = result['ux_fit']
    uy_fit = result['uy_fit']
    A_fit = result['A_fit']
    B_fit = result['B_fit']
    
    # Compute relative errors
    K_I_re = abs(K_I_fit - K_I_true) / abs(K_I_true) * 100.0
    K_II_re = abs(K_II_fit - K_II_true) / abs(K_II_true) * 100.0
    
    # Displacement RMSE
    rmse_ux = np.sqrt(np.mean((ux_fit - ux_clean) ** 2))
    rmse_uy = np.sqrt(np.mean((uy_fit - uy_clean) ** 2))
    rmse_total = np.sqrt(np.mean((ux_fit - ux_clean) ** 2 + (uy_fit - uy_clean) ** 2))
    
    # R² of the fit (against noisy data)
    d = np.concatenate([ux_noisy, uy_noisy])
    d_fit = np.concatenate([ux_fit, uy_fit])
    ss_res = np.sum((d - d_fit) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    
    # Print results
    print("\n" + "=" * 60)
    print("SIF Estimation via Williams Series Fitting")
    print("=" * 60)
    print(f"\nGround-truth Williams coefficients (Mode I):  {coeffs_I_true}")
    print(f"Ground-truth Williams coefficients (Mode II): {coeffs_II_true}")
    print(f"Ground-truth K_I = {K_I_true:.2f} MPa√m")
    print(f"Ground-truth K_II = {K_II_true:.2f} MPa√m")
    print(f"\nSNR: {snr_db:.0f} dB")
    print(f"Number of data points: {len(r_flat)}")
    
    print("\n--- Solving inverse problem (linear least-squares) ---")
    print(f"\nFitted Williams coefficients (Mode I):  {A_fit}")
    print(f"Fitted Williams coefficients (Mode II): {B_fit}")
    print(f"\nFitted K_I  = {K_I_fit:.4f} MPa√m  (true: {K_I_true:.2f})")
    print(f"Fitted K_II = {K_II_fit:.4f} MPa√m  (true: {K_II_true:.2f})")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"K_I  relative error: {K_I_re:.4f} %")
    print(f"K_II relative error: {K_II_re:.4f} %")
    print(f"Displacement RMSE (ux): {rmse_ux:.6e}")
    print(f"Displacement RMSE (uy): {rmse_uy:.6e}")
    print(f"Displacement RMSE (total): {rmse_total:.6e}")
    print(f"R² of fit: {r_squared:.6f}")
    
    # Create metrics dictionary
    metrics = {
        "K_I_true": float(K_I_true),
        "K_II_true": float(K_II_true),
        "K_I_fitted": float(K_I_fit),
        "K_II_fitted": float(K_II_fit),
        "K_I_relative_error_pct": float(K_I_re),
        "K_II_relative_error_pct": float(K_II_re),
        "displacement_rmse_ux": float(rmse_ux),
        "displacement_rmse_uy": float(rmse_uy),
        "displacement_rmse_total": float(rmse_total),
        "R_squared": float(r_squared),
        "SNR_dB": float(snr_db),
        "N_williams_terms": int(N_terms),
        "n_data_points": int(len(r_flat)),
    }
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Save metrics
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to results/metrics.json")
    
    # Save ground truth and reconstruction
    gt_data = np.column_stack([r_flat, theta_flat, ux_clean, uy_clean])
    recon_data = np.column_stack([r_flat, theta_flat, ux_fit, uy_fit])
    np.save('results/ground_truth.npy', gt_data)
    np.save('results/reconstruction.npy', recon_data)
    print("Ground truth saved to results/ground_truth.npy")
    print("Reconstruction saved to results/reconstruction.npy")
    
    # Visualization
    x_cart = r_flat * np.cos(theta_flat) * 1e3  # mm
    y_cart = r_flat * np.sin(theta_flat) * 1e3  # mm
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # (a) GT displacement field (u_x)
    ax = axes[0, 0]
    sc = ax.scatter(x_cart, y_cart, c=ux_clean * 1e6, cmap='RdBu_r', s=15, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='u_x [μm]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('(a) Ground Truth u_x')
    ax.set_aspect('equal')
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
    
    # (b) Fitted displacement field (u_x)
    ax = axes[0, 1]
    sc = ax.scatter(x_cart, y_cart, c=ux_fit * 1e6, cmap='RdBu_r', s=15, edgecolors='none',
                    vmin=ux_clean.min() * 1e6, vmax=ux_clean.max() * 1e6)
    plt.colorbar(sc, ax=ax, label='u_x [μm]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('(b) Fitted u_x (Williams series)')
    ax.set_aspect('equal')
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
    
    # (c) Displacement error map
    ax = axes[1, 0]
    error_ux = (ux_fit - ux_clean) * 1e6
    sc = ax.scatter(x_cart, y_cart, c=error_ux, cmap='coolwarm', s=15, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Δu_x [μm]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'(c) Error map (RMSE={rmse_total*1e6:.3f} μm)')
    ax.set_aspect('equal')
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
    
    # (d) SIF comparison bar chart
    ax = axes[1, 1]
    x_bar = np.arange(2)
    width = 0.3
    gt_vals = [K_I_true, K_II_true]
    fit_vals = [K_I_fit, K_II_fit]
    bars1 = ax.bar(x_bar - width / 2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x_bar + width / 2, fit_vals, width, label='Fitted', color='coral', alpha=0.85)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(['$K_I$', '$K_{II}$'], fontsize=13)
    ax.set_ylabel('SIF [MPa√m]')
    ax.set_title(f'(d) SIF Comparison (K_I RE={K_I_re:.2f}%, K_II RE={K_II_re:.2f}%)')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('SIF Estimation via Williams Series Fitting\n'
                 f'(E={E/1e3:.0f} GPa, ν={nu}, plane stress, {N_terms} terms, SNR={snr_db:.0f} dB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved to results/reconstruction_result.png")
    
    print(f"\n{'='*60}")
    print("DONE — All outputs saved to results/")
    print(f"{'='*60}")
    
    return metrics
