import numpy as np

import matplotlib.pyplot as plt

import mne

def evaluate_results(x_hat, info, evoked, forward, noise_cov):
    """
    Compares standalone results against the reference MNE implementation.
    Generates metrics and plots.
    """
    print("\n=== Phase 3: Reference MNE Reconstruction ===")
    
    # Create inverse operator using MNE (Reference)
    inv_mne = mne.minimum_norm.make_inverse_operator(
        info, forward, noise_cov, 
        loose=0.0, depth=None, fixed=True, verbose=False
    )
    
    # Apply MNE inverse
    stc_mne = mne.minimum_norm.apply_inverse(
        evoked, inv_mne, lambda2=1.0/9.0, method='dSPM', verbose=False
    )
    
    x_mne = stc_mne.data
    
    print("\n=== Phase 4: Evaluation ===")
    print(f"Standalone shape: {x_hat.shape}")
    print(f"MNE shape: {x_mne.shape}")
    
    # Compute Metrics
    mse = np.mean((x_hat - x_mne) ** 2)
    if mse == 0:
        psnr = np.inf
    else:
        psnr = 10 * np.log10(np.max(x_mne)**2 / mse)
    
    corr = np.corrcoef(x_hat.ravel(), x_mne.ravel())[0, 1]
    
    print(f"MSE between Standalone and MNE: {mse:.6e}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Correlation: {corr:.6f}")
    
    if corr > 0.99:
        print("SUCCESS: Standalone implementation matches MNE reference!")
    else:
        print("WARNING: Discrepancy detected.")
        
    # Visualization
    max_idx = np.argmax(np.sum(x_mne**2, axis=1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(evoked.times, x_mne[max_idx], label='MNE Reference', linewidth=2)
    plt.plot(evoked.times, x_hat[max_idx], '--', label='Standalone', linewidth=2)
    plt.title(f'Source Time Course (Vertex {max_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('dSPM value')
    plt.legend()
    plt.grid(True)
    output_img = 'comparison_plot.png'
    plt.savefig(output_img)
    print(f"Comparison plot saved to {output_img}")
