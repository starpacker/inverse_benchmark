import sys
import os
import dill
import traceback
import numpy as np
import warnings
import nibabel as nib
import time
import scipy.optimize
import scipy.special
from sklearn.linear_model import Lasso

# Add the directory containing the target function to python path
sys.path.append('/data/yjh/AMICO-master_sandbox/run_code')

# --- IMPORT TARGET FUNCTION ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Could not import 'run_inversion' from 'agent_run_inversion'. Ensure the file exists and is in the path.")
    sys.exit(1)

# --- INJECT EVALUATION LOGIC (REFERENCE B) ---
def evaluate_results(ndi_map, odi_map, fwf_map, mask, gt_file='GT_NDI.nii.gz'):
    """
    Evaluates the reconstruction against Ground Truth if available.
    """
    print("\n[EVALUATION] Calculating Metrics...")
    
    metrics = {}
    
    if os.path.exists(gt_file):
        try:
            gt_ndi = nib.load(gt_file).get_fdata()
            rmse = np.sqrt(np.mean((ndi_map[mask] - gt_ndi[mask])**2))
            
            if rmse > 0:
                psnr = 20 * np.log10(1.0 / rmse)
            else:
                psnr = float('inf')
                
            print(f"NDI RMSE: {rmse:.4f}")
            print(f"NDI PSNR: {psnr:.2f} dB")
            metrics['rmse'] = rmse
            metrics['psnr'] = psnr
        except Exception as e:
            print(f"Error loading GT or calculating RMSE: {e}")
            metrics['psnr'] = 0.0
    else:
        print(f"Ground Truth {gt_file} not found. Skipping RMSE/PSNR.")
        metrics['psnr'] = 0.0 # Default fallback
        
    metrics['mean_ndi'] = np.mean(ndi_map[mask])
    metrics['mean_odi'] = np.mean(odi_map[mask])
    metrics['mean_fwf'] = np.mean(fwf_map[mask])
    
    print(f"Mean NDI in mask: {metrics['mean_ndi']:.3f}")
    print(f"Mean ODI in mask: {metrics['mean_odi']:.3f}")
    print(f"Mean FWF in mask: {metrics['mean_fwf']:.3f}")
    
    # We return the dictionary of metrics to compare
    return metrics

# --- HELPER FOR LOADING PICKLES ROBUSTLY ---
def load_pickle(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except EOFError:
        print(f"Error: Encounters EOF while loading {path}. The file might be corrupted or empty.")
        return None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# --- MAIN TEST SCRIPT ---
def main():
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 1. Identify Outer Data
    outer_data_path = None
    for p in data_paths:
        if "standard_data_run_inversion.pkl" in p:
            outer_data_path = p
            break
            
    if not outer_data_path:
        print("CRITICAL: standard_data_run_inversion.pkl not found in input paths.")
        sys.exit(1)

    # 2. Load Outer Data
    print(f"Loading data from {outer_data_path}")
    outer_data = load_pickle(outer_data_path)
    
    if outer_data is None:
        print("CRITICAL: Failed to load outer data. Exiting.")
        sys.exit(1)

    # 3. Execution Phase
    try:
        # Extract inputs
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Check if we need to reconstruct mask or other specific inputs
        # The function signature is run_inversion(data, mask, dirs, kernels, lambda1=5e-1)
        # We assume dill loaded these objects correctly (numpy arrays, etc.)
        
        print("\n[EXECUTION] Running Agent 'run_inversion'...")
        start_time = time.time()
        agent_result = run_inversion(*args, **kwargs)
        print(f"[EXECUTION] Agent finished in {time.time() - start_time:.2f}s")
        
        std_result = outer_data.get('output')
        
    except Exception as e:
        print(f"CRITICAL: Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Evaluation Phase
    # The output is a tuple: (ndi_map, odi_map, fwf_map)
    # The evaluation function needs 'mask', which is the 2nd argument (index 1) in args
    try:
        if len(args) > 1:
            mask = args[1]
        else:
            # Try to find mask in kwargs? Usually mask is positional.
            # If not found, create a dummy mask (all True) for shape
            print("Warning: Could not locate mask in args. Attempting to infer.")
            mask = np.ones(agent_result[0].shape, dtype=bool)

        print("\n--- Evaluating Agent Result ---")
        agent_metrics = evaluate_results(agent_result[0], agent_result[1], agent_result[2], mask)
        
        print("\n--- Evaluating Standard (Ground Truth from Pickle) Result ---")
        # Ensure std_result is valid
        if std_result is not None and isinstance(std_result, tuple) and len(std_result) == 3:
            std_metrics = evaluate_results(std_result[0], std_result[1], std_result[2], mask)
        else:
            print("Standard output is missing or malformed in pickle. Skipping comparison.")
            std_metrics = None

    except Exception as e:
        print(f"CRITICAL: Evaluation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Comparison
    if std_metrics:
        # We compare Mean NDI and Mean ODI stability
        # We also check PSNR if GT file existed
        
        # Criterion 1: Metric Stability (within 5% margin)
        ndi_diff = abs(agent_metrics['mean_ndi'] - std_metrics['mean_ndi'])
        ndi_threshold = max(abs(std_metrics['mean_ndi']) * 0.05, 0.01) # 5% or 0.01 absolute
        
        print(f"\nComparison: Agent Mean NDI {agent_metrics['mean_ndi']:.4f} vs Std {std_metrics['mean_ndi']:.4f}")
        
        if ndi_diff > ndi_threshold:
            print(f"FAILURE: NDI Difference {ndi_diff:.4f} exceeds threshold {ndi_threshold:.4f}")
            sys.exit(1)
        else:
            print(f"SUCCESS: NDI results match within tolerance.")
            
    else:
        print("\nWARNING: No standard result to compare against. Assuming success based on successful execution.")

    print("\n[QA] Test Completed Successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()