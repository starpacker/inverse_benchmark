import sys
import os
import dill
import numpy as np
import traceback
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If the agent file is not in the path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# --- Reference B: The Referee (Evaluation Logic) ---
def evaluate_results(reconstruction, gt_images):
    """
    Computes PSNR and SSIM if ground truth is available.
    
    Args:
        reconstruction: (2, H, W) numpy array.
        gt_images: (2, H, W) numpy array or None.
    
    Returns:
        metrics_dict: Dictionary containing average PSNR and SSIM, or None if no GT.
    """
    if gt_images is None:
        print("Ground truth not available for quantitative evaluation.")
        return None

    # Clip negative values
    res_images = np.maximum(reconstruction, 0)
    
    metrics = []
    material_names = ["Bone/Calcium", "Soft Tissue/Water"]
    
    print("\n=== Evaluation ===")
    for i in range(2):
        gt = gt_images[i]
        rec = res_images[i]
        
        # Dynamic range for PSNR
        dmax = np.max(gt) - np.min(gt)
        if dmax == 0: dmax = 1.0
        
        p = psnr(gt, rec, data_range=dmax)
        s = ssim(gt, rec, data_range=dmax)
        metrics.append((p, s))
        
        print(f"Material {i+1} ({material_names[i]}): PSNR = {p:.2f} dB, SSIM = {s:.4f}")
        
    avg_psnr = np.mean([m[0] for m in metrics])
    avg_ssim = np.mean([m[1] for m in metrics])
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return {"psnr": avg_psnr, "ssim": avg_ssim}

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 2. Parse paths to identify execution pattern
    # In this specific case, we have one main file. We check for a closure pattern just in case,
    # but based on the provided input list, it's likely a direct execution or the list is incomplete.
    # We will assume direct execution based on the single file provided in the prompt's input list.
    
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'parent_function' in path:
            inner_paths.append(path)
        else:
            outer_path = path

    if not outer_path:
        print("Error: No outer function data found.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_path}")
    outer_data = load_data(outer_path)
    
    # 3. Execution Phase
    try:
        # Step 1: Run the primary function
        print("Running run_inversion...")
        # Note: args[0] is data, args[1] is space, args[2] is geometry based on the signature
        agent_output = run_inversion(*outer_data['args'], **outer_data['kwargs'])
        
        final_result = None
        std_result = None
        
        # Step 2: Handle results based on execution pattern
        if inner_paths:
            # Pattern 2: Chained Execution (Closure)
            print("Detected Inner/Closure Data. Executing inner function...")
            # We pick the first inner path found for validation (assuming one path of execution)
            inner_path = inner_paths[0] 
            inner_data = load_data(inner_path)
            
            if not callable(agent_output):
                print(f"Error: Expected callable output from outer function for chained execution, got {type(agent_output)}")
                sys.exit(1)
                
            final_result = agent_output(*inner_data['args'], **inner_data['kwargs'])
            std_result = inner_data['output']
            
        else:
            # Pattern 1: Direct Execution
            print("Direct Execution Mode.")
            final_result = agent_output
            std_result = outer_data['output']

        # 4. Evaluation Phase
        # The 'output' in the pickle is the Ground Truth result from the standard run.
        # However, `evaluate_results` expects (reconstruction, gt_images).
        # In this specific context (CT reconstruction), the `std_result` (the output of the function)
        # IS the reconstruction. To evaluate "integrity" vs Ground Truth, we usually need the true Phantom.
        # IF the pickle does not contain the true phantom, we can only compare Agent Result vs Standard Result
        # to ensure the code hasn't broken/diverged.
        
        # In optimization problems stored via dill/pickle, `std_result` is often the "Gold Standard" 
        # computed by the original code. We treat `std_result` as the reference for `evaluate_results` 
        # to see if our current run matches the previous valid run, OR if the input args contained GT.
        
        # Let's inspect the args for GT. Usually in CT, GT is not passed to the solver.
        # So we compare Current Run (Agent) vs Previous Run (Standard Result).
        
        print("\nComparing Agent Output vs Standard (Pickled) Output...")
        
        # Since evaluate_results calculates PSNR/SSIM, passing (agent, std) treats std as the Ground Truth.
        scores = evaluate_results(final_result, std_result)

        if scores is None:
            print("Evaluation returned None. Cannot verify performance integrity quantitatively.")
            # If visual/quantitative check fails, we might fall back to basic shape check
            if final_result.shape == std_result.shape:
                print("Shapes match. Assuming pass due to lack of GT for metrics.")
                sys.exit(0)
            else:
                print("Shapes mismatch!")
                sys.exit(1)
        
        # 5. Verification & Reporting
        # High PSNR/SSIM means the current run is close to the recorded standard run.
        # Since both are reconstruction algorithms, if we get high similarity, the refactoring is valid.
        
        # Thresholds: Since we are likely comparing float calculations on potentially different hardware (GPU),
        # we allow some deviation.
        # PSNR > 40dB is typically "indistinguishable".
        # PSNR > 100dB implies almost bitwise equality.
        # If the code is deterministic and hardware is same, we expect very high PSNR.
        # If the code involves random initialization (it does not seem to, FBP init), it should be stable.
        
        psnr_val = scores['psnr']
        ssim_val = scores['ssim']
        
        # 30dB is a safe lower bound for "functionally equivalent" reconstruction updates.
        # 0.9 SSIM is a safe lower bound.
        
        if psnr_val > 30.0 and ssim_val > 0.9:
            print(f"\nSUCCESS: Performance integrity verified. PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
            sys.exit(0)
        else:
            print(f"\nFAILURE: Performance degraded or diverged too much. PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()