import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def photoelectric_cross_section(E_keV):
    """Approximate photoelectric absorption cross-section in cm^2.
    Simplified Morrison & McCammon approximation."""
    return 2.0e-22 * (E_keV) ** (-8.0/3.0)

def absorbed_powerlaw(E_keV, gamma, K, N_H_1e22):
    """Absorbed power-law X-ray spectrum.
    Args:
        E_keV: energy bins in keV
        gamma: photon index (typically 1.5-3.0)
        K: normalization (photons/cm2/s/keV at 1 keV)
        N_H_1e22: hydrogen column density in units of 10^22 cm^-2
    Returns:
        photon flux in photons/cm2/s/keV
    """
    sigma = photoelectric_cross_section(E_keV)
    absorption = np.exp(-N_H_1e22 * 1e22 * sigma)
    return K * E_keV**(-gamma) * absorption

def evaluate_results(data_dict, result_dict):
    """Evaluate reconstruction quality and save results.
    
    Args:
        data_dict: dictionary from load_and_preprocess_data
        result_dict: dictionary from run_inversion
        
    Returns:
        metrics: dictionary containing evaluation metrics
    """
    E = data_dict['E_centers']
    dE = data_dict['dE']
    observed = data_dict['observed']
    expected = data_dict['expected_counts']
    background = data_dict['background']
    true_params = data_dict['true_params']
    
    gamma_fit = result_dict['gamma_fit']
    K_fit = result_dict['K_fit']
    NH_fit = result_dict['NH_fit']
    recovered = result_dict['recovered_counts']
    
    # True total counts (source + background)
    true_total = expected + background
    
    # Compute metrics
    psnr_val = 10 * np.log10(np.max(true_total)**2 / np.mean((true_total - recovered)**2))
    cc = np.corrcoef(true_total, recovered)[0, 1]
    rmse = np.sqrt(np.mean((true_total - recovered)**2))
    
    # Parameter errors
    param_errors = {
        'gamma': {
            'true': true_params['gamma'], 
            'fitted': gamma_fit,
            'rel_error': abs(gamma_fit - true_params['gamma']) / true_params['gamma']
        },
        'K': {
            'true': true_params['K'], 
            'fitted': K_fit,
            'rel_error': abs(K_fit - true_params['K']) / true_params['K']
        },
        'N_H': {
            'true': true_params['N_H'], 
            'fitted': NH_fit,
            'rel_error': abs(NH_fit - true_params['N_H']) / true_params['N_H']
        }
    }
    mean_rel_error = np.mean([v['rel_error'] for v in param_errors.values()])
    
    # Build metrics dictionary
    metrics = {
        'task': 'bxa_xray',
        'psnr_db': float(psnr_val),
        'correlation_coefficient': float(cc),
        'rmse_counts': float(rmse),
        'mean_parameter_relative_error': float(mean_rel_error),
        'parameters': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in param_errors.items()}
    }
    
    # Save metrics to JSON
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print evaluation results
    print(f"[EVAL] PSNR = {psnr_val:.2f} dB, CC = {cc:.6f}, RMSE = {rmse:.2f}")
    for k, v in param_errors.items():
        print(f"  {k}: true={v['true']}, fitted={v['fitted']:.6f}, error={v['rel_error']*100:.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Observed data
    axes[0, 0].step(E, observed, 'b-', lw=0.5, label='Observed')
    axes[0, 0].plot(E, true_total, 'r-', lw=2, label='True model')
    axes[0, 0].set_xlabel('Energy (keV)')
    axes[0, 0].set_ylabel('Counts')
    axes[0, 0].set_title('X-ray Spectrum (Data)')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # Plot 2: True vs Recovered
    axes[0, 1].plot(E, true_total, 'r-', lw=2, label='True')
    axes[0, 1].plot(E, recovered, 'g--', lw=2, label='Recovered')
    axes[0, 1].set_xlabel('Energy (keV)')
    axes[0, 1].set_ylabel('Counts')
    axes[0, 1].set_title('True vs Recovered')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')

    # Plot 3: Residuals
    residual = (observed - recovered) / np.sqrt(np.maximum(recovered, 1))
    axes[0, 2].step(E, residual, 'k-', lw=0.5)
    axes[0, 2].axhline(0, color='r', ls='--')
    axes[0, 2].set_xlabel('Energy (keV)')
    axes[0, 2].set_ylabel('χ residual')
    axes[0, 2].set_title('Residuals')

    # Plot 4: Parameter comparison bars
    names = list(param_errors.keys())
    true_vals = [param_errors[n]['true'] for n in names]
    fit_vals = [param_errors[n]['fitted'] for n in names]
    x = np.arange(len(names))
    axes[1, 0].bar(x - 0.15, true_vals, 0.3, label='True', color='blue', alpha=0.7)
    axes[1, 0].bar(x + 0.15, fit_vals, 0.3, label='Fitted', color='orange', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_title('Parameter Recovery')
    axes[1, 0].legend()

    # Plot 5: Unfolded spectrum (in flux units)
    true_flux = absorbed_powerlaw(E, true_params['gamma'], true_params['K'], true_params['N_H'])
    fit_flux = absorbed_powerlaw(E, gamma_fit, K_fit, NH_fit)
    axes[1, 1].loglog(E, E**2 * true_flux, 'r-', lw=2, label='True')
    axes[1, 1].loglog(E, E**2 * fit_flux, 'g--', lw=2, label='Recovered')
    axes[1, 1].set_xlabel('Energy (keV)')
    axes[1, 1].set_ylabel('E²F(E)')
    axes[1, 1].set_title('Unfolded Spectrum (EFE)')
    axes[1, 1].legend()

    # Plot 6: Error text summary
    axes[1, 2].axis('off')
    text = f"Spectral Fitting Results\n{'='*30}\n"
    for n in names:
        text += f"{n}: {param_errors[n]['true']} → {param_errors[n]['fitted']:.5f} ({param_errors[n]['rel_error']*100:.1f}%)\n"
    text += f"\nPSNR = {psnr_val:.2f} dB\nCC = {cc:.6f}\nMean Param Error = {mean_rel_error*100:.2f}%"
    axes[1, 2].text(0.1, 0.5, text, fontsize=12, family='monospace', va='center', transform=axes[1, 2].transAxes)

    plt.suptitle(f"X-ray Spectral Fitting | PSNR={psnr_val:.2f} dB | CC={cc:.6f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save arrays
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), true_total)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), recovered)
    
    return metrics
