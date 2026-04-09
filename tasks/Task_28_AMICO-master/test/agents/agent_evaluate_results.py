import os

import warnings

import numpy as np

import nibabel as nib

warnings.filterwarnings("ignore")

def evaluate_results(ndi_map, odi_map, fwf_map, mask, gt_file='GT_NDI.nii.gz'):
    """
    Evaluates the reconstruction against Ground Truth if available.
    """
    print("\n[EVALUATION] Calculating Metrics...")
    
    if os.path.exists(gt_file):
        gt_ndi = nib.load(gt_file).get_fdata()
        rmse = np.sqrt(np.mean((ndi_map[mask] - gt_ndi[mask])**2))
        
        if rmse > 0:
            psnr = 20 * np.log10(1.0 / rmse)
        else:
            psnr = float('inf')
            
        print(f"NDI RMSE: {rmse:.4f}")
        print(f"NDI PSNR: {psnr:.2f} dB")
    else:
        print(f"Ground Truth {gt_file} not found. Skipping RMSE/PSNR.")
        
    print(f"Mean NDI in mask: {np.mean(ndi_map[mask]):.3f}")
    print(f"Mean ODI in mask: {np.mean(odi_map[mask]):.3f}")
    print(f"Mean FWF in mask: {np.mean(fwf_map[mask]):.3f}")
    
    return True
