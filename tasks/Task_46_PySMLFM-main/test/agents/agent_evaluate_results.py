import numpy as np

import numpy.typing as npt

def evaluate_results(gt_points: npt.NDArray[float], reconstructed_points: npt.NDArray[float]) -> float:
    """
    Evaluates the reconstruction by matching points to ground truth and calculating RMSE.
    Returns RMSE.
    """
    print("\n=== RECONSTRUCTION RESULTS ===")
    print(f"Reconstructed {len(reconstructed_points)} points.")
    
    if len(reconstructed_points) == 0:
        print("No points reconstructed.")
        return float('inf')

    rec_xyz = reconstructed_points[:, 0:3]
    mse_sum = 0.0
    matches = 0
    
    print("\nComparison (GT vs Rec):")
    for gt in gt_points:
        dists = np.sqrt(np.sum((rec_xyz - gt)**2, axis=1))
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist < 1.0: # Match threshold 1 micron
            rec = rec_xyz[min_dist_idx]
            print(f"GT: {gt} -> Rec: {rec} (Err: {min_dist:.4f} um)")
            mse_sum += min_dist**2
            matches += 1
        else:
            print(f"GT: {gt} -> No match found (Min dist: {min_dist:.4f} um)")
    
    if matches > 0:
        rmse = np.sqrt(mse_sum / matches)
        print(f"\nRMSE (matched points): {rmse:.4f} microns")
        print(f"PSNR (proxy): {20 * np.log10(10.0 / rmse):.2f} dB (assuming peak=10um)")
        return rmse
    else:
        print("No matches found.")
        return float('inf')
