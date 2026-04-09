import math

import numpy as np

def evaluate_results(points_3d: np.ndarray, expected_depth: float = 5000.0) -> None:
    """
    Calculates statistics on reconstructed point cloud and compares to theoretical model.
    """
    if points_3d.shape[0] == 0:
        print("EVALUATION FAILED: No 3D points reconstructed.")
        return

    z_vals = points_3d[:, 2]
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    
    error = abs(z_mean - expected_depth)
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Reconstructed Points: {points_3d.shape[0]}")
    print(f"Mean Z Depth:         {z_mean:.4f} mm")
    print(f"Std Dev Z:            {z_std:.4f} mm")
    print(f"Theoretical Z:        {expected_depth:.4f} mm")
    print(f"Absolute Error:       {error:.4f} mm")
    
    # Tolerance check
    if error < 50.0: # Generous tolerance for discrete simulation
        print("STATUS: SUCCESS (Within tolerance)")
    else:
        print("STATUS: WARNING (High deviation)")
        
    # Calculate simple PSNR proxy (Linearity check logic from original code)
    # Since we don't have the phase map here, we check planarity of Z
    # Ideal surface is flat (Z = constant)
    mse = np.mean((z_vals - z_mean)**2)
    if mse == 0:
        psnr = 100
    else:
        max_val = np.max(z_vals)
        psnr = 20 * math.log10(max_val / math.sqrt(mse))
    
    print(f"Surface Planarity PSNR: {psnr:.2f} dB")
    print("="*30 + "\n")
