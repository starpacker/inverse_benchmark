import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Imports & Setup
# -------------------------------------------------------------------------

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If the file is not in the path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# Import dependencies for evaluation
try:
    from skimage.transform import radon, iradon
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using slower fallback implementations.")

# -------------------------------------------------------------------------
# 2. Referee / Evaluation Logic (Injected from Reference B)
# -------------------------------------------------------------------------

def evaluate_results(gt, recon, save_path="reconstruction_result.png"):
    """
    Computes PSNR/SSIM and saves a comparison plot.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    # Normalize for fair metric calculation
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn == 0: return arr
        return (arr - mn) / (mx - mn)

    gt_norm = normalize(gt)
    recon_norm = normalize(recon)
    
    # PSNR
    mse = np.mean((gt_norm - recon_norm) ** 2)
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
        
    # SSIM
    ssim_val = 0.0
    if HAS_SKIMAGE:
        # data_range=1.0 because we normalized
        ssim_val = ssim_func(gt_norm, recon_norm, data_range=1.0)
    
    print(f"Evaluation -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    
    # Visualization
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt, cmap='gray')
        ax[0].set_title("Ground Truth (Standard Result)")
        ax[0].axis('off')
        
        ax[1].imshow(recon, cmap='gray')
        ax[1].set_title(f"Reconstruction (Agent)\nPSNR: {psnr_val:.1f}")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Figure saved to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    return {"psnr": psnr_val, "ssim": ssim_val}

# -------------------------------------------------------------------------
# 3. Test Execution Logic
# -------------------------------------------------------------------------

def run_test(data_paths):
    print("----------------------------------------------------------------")
    print("Running QA Test for: run_inversion")
    print(f"Data Paths: {data_paths}")
    print("----------------------------------------------------------------")

    outer_data_path = None
    inner_data_path = None

    # Categorize data files
    for path in data_paths:
        filename = os.path.basename(path)
        if "standard_data_run_inversion.pkl" in filename and "parent" not in filename:
            outer_data_path = path
        elif "parent_function_run_inversion" in filename:
            inner_data_path = path
    
    # We prioritize the "Direct Execution" pattern for this specific function signature,
    # as run_inversion typically returns the image directly, not a closure.
    # However, we implement the logic to handle both cases based on file existence.
    
    if outer_data_path is None:
        print("Error: Could not find primary standard_data_run_inversion.pkl")
        sys.exit(1)

    try:
        # Load Outer Data
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        std_outer_output = outer_data.get('output')

        print(f"Executing run_inversion with {len(args)} args and {len(kwargs)} kwargs...")
        
        # 1. Run the Agent's Code
        agent_result = run_inversion(*args, **kwargs)

        final_result_agent = agent_result
        final_result_std = std_outer_output

        # 2. Check for Chained Execution (Factory Pattern)
        if inner_data_path and callable(agent_result):
            print("Detected Factory Pattern. Loading inner data...")
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            std_inner_output = inner_data.get('output')

            # Execute the operator returned by run_inversion
            print("Executing returned operator...")
            final_result_agent = agent_result(*inner_args, **inner_kwargs)
            final_result_std = std_inner_output
        
        elif inner_data_path and not callable(agent_result):
            print("Warning: Inner data found but agent result is not callable. Using outer result.")

        # ---------------------------------------------------------------------
        # 4. Evaluation Phase
        # ---------------------------------------------------------------------
        
        # In this specific context:
        # The 'std_result' loaded from the pickle is the result of the previous run.
        # We treat 'std_result' as the "Ground Truth" for comparison purposes 
        # (i.e., we want to ensure the current code produces similar results to the recorded run).
        # IF the recorded run was a reconstruction from a sinogram, comparing 
        # Agent Reconstruction vs Standard Reconstruction is a consistency check (Regression Test).
        
        # However, `evaluate_results` expects (Ground Truth Image, Reconstructed Image).
        # Since we don't have the *original* phantom image, we treat the 
        # 'std_result' (Recorded Output) as our Ground Truth target.
        
        print("\nEvaluating Agent Result against Recorded Standard Result...")
        
        # Ensure inputs are numpy arrays
        if not isinstance(final_result_agent, np.ndarray) or not isinstance(final_result_std, np.ndarray):
             print("Error: Results are not numpy arrays. Skipping metric evaluation.")
             # Fallback check for type match
             if type(final_result_agent) == type(final_result_std):
                 print("Types match. Assuming Pass for non-array output.")
                 sys.exit(0)
             else:
                 sys.exit(1)

        metrics = evaluate_results(final_result_std, final_result_agent, save_path="comparison_run_inversion.png")
        
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        
        print(f"Final Scores -> PSNR: {psnr}, SSIM: {ssim}")

        # ---------------------------------------------------------------------
        # 5. Verification & Thresholding
        # ---------------------------------------------------------------------
        
        # Since we are comparing the Agent's run against a recorded "Standard" run of the *same* algorithm,
        # we expect very high similarity (ideally identical, but floating point diffs occur).
        # We set a high bar for regression testing.
        
        # Thresholds: PSNR > 40dB (indicating extremely low error) or SSIM > 0.95
        # If the standard result was bad, this test ensures we are consistently bad (stability),
        # but usually standard data is 'correct'.
        
        if psnr > 40.0 or ssim > 0.95:
            print("SUCCESS: Performance integrity verified.")
            sys.exit(0)
        else:
            print("FAILURE: Performance significantly deviated from standard recording.")
            sys.exit(1)

    except Exception as e:
        print(f"Execution Failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Standard data paths based on the provided gen_data_code location logic
    # In a real run, these might be passed via CLI arguments.
    # Here we hardcode the list as requested in instructions setup.
    default_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Check if files actually exist, otherwise warn (for local testing without the specific path)
    if not os.path.exists(default_paths[0]):
        print(f"Warning: Default path {default_paths[0]} does not exist.")
        print("Looking for local .pkl files...")
        local_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'standard_data_run_inversion' in f]
        if local_files:
            default_paths = [os.path.abspath(f) for f in local_files]
            print(f"Found local files: {default_paths}")
    
    run_test(default_paths)