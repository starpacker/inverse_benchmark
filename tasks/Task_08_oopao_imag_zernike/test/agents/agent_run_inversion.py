import numpy as np

def run_inversion(data_dict):
    """
    Performs iterative Zernike decomposition of atmospheric turbulence.
    
    The inverse model decomposes the OPD map into Zernike coefficients:
    c = Z^+ * phi
    
    Then reconstructs: opd_rec = Z * c
    
    Args:
        data_dict: Dictionary containing telescope, atmosphere, Zernike basis, etc.
        
    Returns:
        results_dict: Dictionary containing inversion results
    """
    print("\n[4] Inverse Model: Decomposing Atmospheric Turbulence...")
    
    tel = data_dict['telescope']
    atm = data_dict['atmosphere']
    Z = data_dict['zernike_object']
    Z_inv = data_dict['zernike_inverse']
    n_iter = data_dict['n_iterations']
    
    rmse_history = []
    all_coeffs = []
    all_opd_original = []
    all_opd_reconstructed = []
    
    for i in range(n_iter):
        # Update atmosphere
        atm.update()
        current_opd = atm.OPD.copy()
        
        # Explicit Decomposition Step
        # Extract valid phase points inside pupil
        opd_masked = current_opd[np.where(tel.pupil == 1)]
        
        # Project onto Zernike Basis (Least Squares fitting)
        # c = Z_dagger * opd
        coeffs = Z_inv @ opd_masked
        
        # Reconstruction
        # opd_rec = Z * c
        reconstructed_opd = np.squeeze(Z.modesFullRes @ coeffs)
        
        # Error calculation
        diff = (current_opd - reconstructed_opd) * tel.pupil
        rmse = np.std(diff[tel.pupil == 1])
        rmse_history.append(rmse)
        
        # Store results
        all_coeffs.append(coeffs.copy())
        all_opd_original.append(current_opd.copy())
        all_opd_reconstructed.append(reconstructed_opd.copy())
        
        print(f"    Iter {i+1}: Fitting RMSE = {rmse * 1e9:.1f} nm")
    
    results_dict = {
        'rmse_history': np.array(rmse_history),
        'all_coeffs': all_coeffs,
        'all_opd_original': all_opd_original,
        'all_opd_reconstructed': all_opd_reconstructed,
        'final_rmse': rmse_history[-1],
        'mean_rmse': np.mean(rmse_history),
    }
    
    return results_dict