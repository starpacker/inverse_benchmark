import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add the target directory to sys.path to import the agent function
# Assuming the agent code is in the current directory or a known relative path
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If strictly local, try direct import
    try:
        from agent_run_inversion import run_inversion
    except ImportError:
        print("Could not import run_inversion. Ensure agent_run_inversion.py is in the path.")
        sys.exit(1)

# --- INJECTED REFEREE FUNCTION (Reference B) ---
def evaluate_results(ground_truth, recon_ph, data_ISM_noise):
    """
    Calculates PSNR/SSIM and saves result plot.
    RETURNS: (psnr_val, ssim_val) tuple for programmatic comparison.
    """
    print("Evaluating Results...")
    
    # Normalize images for metrics calculation
    # Ensure inputs are numpy arrays
    if hasattr(ground_truth, 'detach'): ground_truth = ground_truth.detach().cpu().numpy()
    if hasattr(recon_ph, 'detach'): recon_ph = recon_ph.detach().cpu().numpy()
    if hasattr(data_ISM_noise, 'detach'): data_ISM_noise = data_ISM_noise.detach().cpu().numpy()

    # Handle shapes. Ground truth might be (D, H, W) or (H, W).
    # We usually evaluate on the central slice or the first slice if 3D, or the image itself if 2D.
    
    def get_norm_image(img):
        if img.ndim == 3:
            res = img[0] # Take first slice for evaluation
        else:
            res = img
        # Avoid division by zero
        mx = res.max()
        if mx == 0: return res
        return res / mx

    def get_ism_sum_norm(img):
        # ISM data is often (D, H, W, GridX, GridY) or (H, W, Grid) -> Sum over last dimension
        # Based on context: data_ISM_noise.sum(-1)
        res = img.sum(-1)
        mx = res.max()
        if mx == 0: return res
        return res / mx

    gt_norm = get_norm_image(ground_truth)
    recon_norm = get_norm_image(recon_ph)
    
    # Calculate PSNR and SSIM
    psnr_val = psnr(gt_norm, recon_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Reconstruction (In-Focus) vs Ground Truth:")
    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")

    # Optional: Compare Raw ISM if dimensions allow
    try:
        ism_norm = get_ism_sum_norm(data_ISM_noise)
        # Ensure dimensions match for metric calc
        if gt_norm.shape == ism_norm.shape:
            psnr_ism = psnr(gt_norm, ism_norm, data_range=1.0)
            ssim_ism = ssim(gt_norm, ism_norm, data_range=1.0)
            print(f"Raw ISM Sum (Confocal-like) vs Ground Truth:")
            print(f"PSNR: {psnr_ism:.4f}")
            print(f"SSIM: {ssim_ism:.4f}")
    except Exception as e:
        print(f"Skipping Raw ISM comparison due to shape mismatch or error: {e}")

    # Plot results
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_norm, cmap='magma')
        axes[0].set_title('Ground Truth (In-Focus)')
        
        # Try to plot ISM sum, handle if not available
        try:
            ism_disp = data_ISM_noise.sum(-1)
            if ism_disp.ndim == 3: ism_disp = ism_disp[0]
            axes[1].imshow(ism_disp, cmap='magma')
        except:
            axes[1].text(0.5, 0.5, 'ISM Data Unavailable', ha='center')
        axes[1].set_title('Raw ISM Sum')
        
        axes[2].imshow(recon_norm, cmap='magma')
        axes[2].set_title('s2ISM Reconstruction')
        plt.tight_layout()
        plt.savefig('s2ism_result.png')
        print("Result image saved to s2ism_result.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return psnr_val, ssim_val

# --- HELPER INJECTION ---
# Ensure partial_convolution_rfft etc are available if they were pickled as global references.
# In this specific task, run_inversion is self-contained in the agent file, 
# but dill loading might require classes/functions to be defined if they were saved by reference.
# We trust the provided agent code imports them or defines them.

def main():
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Analyze paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if "parent_function" in p:
            inner_paths.append(p)
        elif "standard_data_run_inversion.pkl" in os.path.basename(p):
            outer_path = p
    
    if not outer_path:
        print("No standard_data_run_inversion.pkl found. Aborting.")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    # Prepare inputs for run_inversion
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    
    # IMPORTANT: The provided context suggests run_inversion(data_ISM_noise, Psf).
    # However, the evaluation function needs Ground Truth to calculate metrics.
    # The 'standard_data' capture usually captures inputs and outputs of the function call.
    # It DOES NOT usually capture the Ground Truth unless it was passed as an argument.
    # Looking at the reference code `run_inversion(data_ISM_noise, Psf)`, Ground Truth is NOT an input.
    # Therefore, we cannot strictly run `evaluate_results` which requires GT.
    #
    # HEURISTIC: 
    # 1. We check if the expected output (from pkl) exists.
    # 2. We treat the EXPECTED OUTPUT from the .pkl as the "Ground Truth" or "Golden Reference" 
    #    for the purpose of verifying regression, OR we simply compare Agent Output vs Stored Output.
    # 3. BUT, the instructions say: "use a Standard Evaluation Function (evaluate_results) to compare the Agent's quality against the Ground Truth."
    # 
    # If the Ground Truth is not in inputs, we assume the user might have provided it 
    # or the standard output *is* the reconstruction, and we compare Agent Reconstruction vs Standard Reconstruction to ensure they match (integrity).
    #
    # However, `evaluate_results(ground_truth, recon_ph, data_ISM_noise)` implies we have GT.
    # If we don't have GT, we will use the Agent's result as 'recon_ph' and the Stored result as 'ground_truth' 
    # to see how close the new run is to the old successful run (Regression Testing).
    #
    # Let's extract arguments:
    # run_inversion(data_ISM_noise, Psf)
    
    if len(outer_args) >= 1:
        data_ism_noise = outer_args[0]
    elif 'data_ISM_noise' in outer_kwargs:
        data_ism_noise = outer_kwargs['data_ISM_noise']
    else:
        # Fallback if args not captured clearly
        print("Could not locate data_ISM_noise input.")
        sys.exit(1)

    # Execute Agent
    print("Executing run_inversion (Agent)...")
    try:
        agent_result = run_inversion(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Agent Execution Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Standard (Expected) Result
    std_result = outer_data.get('output')

    # Since we are doing QA on an Inversion Algorithm, the "Ground Truth" (the real object) 
    # is usually needed for PSNR/SSIM. The .pkl captures inputs to the function.
    # If the function is `run_inversion(data, psf)`, the GT is not here.
    #
    # ADAPTATION:
    # We will use the Stored Output (which is a valid reconstruction) as the "Ground Truth" for the sake of the metric,
    # OR we treat this as a pure regression test: New Result vs Old Result.
    # The `evaluate_results` function prints PSNR/SSIM. 
    # If we pass (std_result, agent_result, data_ism_noise), we measure how similar the agent is to the standard.
    # This yields PSNR infinity if they are identical.
    
    print("\n--- QA Validation ---")
    print("Scenario: Comparing Agent Output vs Stored 'Golden' Output (Regression Test)")
    
    # We use the provided evaluate_results.
    # signature: evaluate_results(ground_truth, recon_ph, data_ISM_noise)
    # We treat 'std_result' as the Ground Truth reference for this regression test.
    
    try:
        # We assume std_result is the "good" reconstruction from a previous run.
        # agent_result is the current reconstruction.
        psnr_agent, ssim_agent = evaluate_results(std_result, agent_result, data_ism_noise)
        
        # Since we are comparing against the previous output of the same algorithm, 
        # we expect very high similarity (near identical).
        
        print(f"\nMetric Comparison (Agent vs Stored Result):")
        print(f"PSNR: {psnr_agent}")
        print(f"SSIM: {ssim_agent}")
        
        # Criteria: High similarity to ensure integrity
        # PSNR > 40 usually implies excellent fidelity. > 30 is good. 
        # Since this is deterministic code (seeded), it should be very high.
        
        if psnr_agent < 30.0 or ssim_agent < 0.95:
            print("FAILURE: Agent result deviates significantly from standard result.")
            sys.exit(1)
        else:
            print("SUCCESS: Agent result matches standard integrity.")
            sys.exit(0)

    except Exception as e:
        print(f"Evaluation Logic Failed: {e}")
        traceback.print_exc()
        # Fallback to simple equality check if plotting/metrics fail due to shape weirdness
        if np.allclose(std_result, agent_result, rtol=1e-2, atol=1e-2):
             print("SUCCESS: Numpy array match (Fallback).")
             sys.exit(0)
        sys.exit(1)

if __name__ == "__main__":
    main()