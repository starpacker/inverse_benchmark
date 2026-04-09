import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Import Target Function ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Could not import 'run_inversion' from 'agent_run_inversion.py'. Ensure the file is in the python path.")
    sys.exit(1)

# --- Referee Function (Reference B) ---
def evaluate_results(recon, gt, result_name, output_dir="."):
    """
    Computes metrics if GT is available, and saves the image.
    Returns the PSNR value for programmatic comparison.
    """
    def min_max_scale(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    recon_norm = min_max_scale(recon)
    
    val_psnr = 0.0
    metrics_str = ""
    if gt is not None:
        gt_norm = min_max_scale(gt)
        # Use data_range=1.0 since we min-max scaled to [0,1]
        val_psnr = psnr(gt_norm, recon_norm, data_range=1.0)
        val_ssim = ssim(gt_norm, recon_norm, data_range=1.0)
        metrics_str = f"PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}"
        print(f"Evaluation for {result_name}: {metrics_str}")
    
    # Save Image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.figure()
    plt.imshow(recon_norm, cmap='gray')
    title = f"{result_name}"
    if metrics_str:
        title += f"\n{metrics_str}"
    plt.title(title)
    plt.axis('off')
    
    out_path = os.path.join(output_dir, f"{result_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved result to {out_path}")
    
    return val_psnr

# --- Main Test Logic ---
def main():
    # 1. Configuration
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    output_dir = "test_results"
    
    # Identify files
    outer_file = None
    inner_files = []
    
    for path in data_paths:
        if "parent_function" in path:
            inner_files.append(path)
        else:
            outer_file = path

    if not outer_file:
        print("Error: No primary outer data file found.")
        sys.exit(1)

    print(f"Loading outer data from {outer_file}...")
    try:
        with open(outer_file, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Execution
    try:
        # Step A: Run Main Function
        print("Running 'run_inversion' with outer data...")
        agent_output = run_inversion(*outer_data['args'], **outer_data['kwargs'])
        
        # Step B: Handle Chained vs Direct Execution
        final_agent_result = None
        standard_gt = None
        
        if inner_files:
            # Pattern 2: Chained Execution
            # If inner files exist, the outer function returns a callable (operator)
            if not callable(agent_output):
                print(f"Error: Expected callable output for chained execution, got {type(agent_output)}.")
                sys.exit(1)
            
            # For simplicity, we process the first inner file found (usually there's a sequence, but unit testing one is sufficient for logic check)
            inner_path = inner_files[0]
            print(f"Loading inner data from {inner_path}...")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            print("Running inner operator...")
            final_agent_result = agent_output(*inner_data['args'], **inner_data['kwargs'])
            standard_gt = inner_data['output'] # The expected result recorded in the pickle
            
        else:
            # Pattern 1: Direct Execution
            final_agent_result = agent_output
            standard_gt = outer_data['output']

        # 3. Evaluation
        print("\n--- Evaluation Phase ---")
        
        # We need a Ground Truth image to calculate PSNR. 
        # In the provided 'run_inversion', the output is a reconstruction.
        # Ideally, we compare the Agent's Reconstruction vs the Standard (recorded) Reconstruction.
        # This ensures the code hasn't broken.
        
        # Evaluate Agent Result (comparing against Standard Output as "GT" for regression testing)
        score_agent_vs_std = evaluate_results(final_agent_result, standard_gt, "Agent_Reconstruction_vs_Recorded", output_dir)
        
        # Also visualize the Standard Result alone for reference
        evaluate_results(standard_gt, None, "Recorded_Standard_Output", output_dir)

        # 4. Verification
        # Since we are comparing the current run against a previous recorded run of the same algorithm,
        # the results should be nearly identical (PSNR -> Infinity).
        # However, due to floating point differences or random seeds, we expect high similarity.
        
        print(f"\nMetric (PSNR of Agent vs Recorded): {score_agent_vs_std:.2f}")

        # Thresholds
        # If the code logic is identical, PSNR should be very high (> 50 or 100).
        # If it's an optimization process, minor drift is allowed.
        PASS_THRESHOLD_PSNR = 40.0 
        
        if score_agent_vs_std >= PASS_THRESHOLD_PSNR:
            print("SUCCESS: Performance integrity verified.")
            sys.exit(0)
        else:
            print("FAILURE: Performance significantly degraded or logic altered.")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()