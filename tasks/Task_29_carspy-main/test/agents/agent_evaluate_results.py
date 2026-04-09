import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

_HAS_MATPLOTLIB = True

def evaluate_results(y_true, y_pred, nu_axis, params_pred, params_true=None):
    """
    Calculates metrics and plots results.
    
    Args:
        y_true (np.ndarray): Measured spectrum.
        y_pred (np.ndarray): Fitted spectrum.
        nu_axis (np.ndarray): Wavenumber axis.
        params_pred (dict): Retrieved parameters.
        params_true (dict, optional): Ground truth parameters.
    """
    # 1. MSE
    mse = np.mean((y_true - y_pred)**2)
    print(f"MSE: {mse:.6e}")
    
    # 2. PSNR
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse) # Signal peak is 1.0
        print(f"PSNR: {psnr:.2f} dB")
    
    # 3. Parameter Error
    if params_true:
        T_true = params_true['temperature']
        T_pred = params_pred['temperature']
        err = abs(T_true - T_pred)
        print(f"Temperature Error: {err:.2f} K (True: {T_true} K, Pred: {T_pred:.2f} K)")
        
    # 4. Plotting
    if _HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(nu_axis, y_true, 'k.', label='Measured')
        plt.plot(nu_axis, y_pred, 'r-', linewidth=2, label=f'Fit (T={params_pred["temperature"]:.0f}K)')
        plt.xlabel('Wavenumber (cm-1)')
        plt.ylabel('Normalized Intensity')
        plt.title('CARS Inversion Result')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('inversion_result.png')
        print("Plot saved to 'inversion_result.png'")
