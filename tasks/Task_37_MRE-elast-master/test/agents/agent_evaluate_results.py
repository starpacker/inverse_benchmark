import numpy as np

from scipy.interpolate import griddata

import matplotlib.pyplot as plt

def evaluate_results(E_true, E_recon, vertices):
    """
    Calculates CNR and RMSE, and plots the results.
    """
    # Calculate Metrics
    thresh = 0.8 * np.max(E_recon)
    idxe = np.where(E_recon > thresh)
    idxb = np.where(E_recon < thresh)
    
    EE = E_recon[idxe[0]]
    BB = E_recon[idxb[0]]
    
    if len(EE) == 0 or len(BB) == 0:
        cnr = 0
    else:
        cnr = 10 * np.log10(2 * (np.mean(EE) - np.mean(BB))**2 / (np.var(EE) + np.var(BB)))
        
    rms = np.sqrt(np.mean(np.abs(2 * (E_recon - E_true) / (E_true + E_recon + 1e-9))**2))
    
    print("\nEvaluation Results:")
    print(f"CNR: {cnr:.2f} dB")
    print(f"RMSE: {rms:.4f}")

    # Plotting
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_new, y_new)
    
    E_true_im = griddata((x, y), E_true, (X, Y), method='linear')
    E_recon_im = griddata((x, y), E_recon, (X, Y), method='linear')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(E_true_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    plt.title("Ground Truth Stiffness")
    
    plt.subplot(1, 2, 2)
    plt.imshow(E_recon_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    plt.title(f"Reconstructed (CNR={cnr:.1f}dB)")
    
    plt.savefig('mre_refactored_result.png')
    print("Result saved to mre_refactored_result.png")
    
    return cnr, rms
