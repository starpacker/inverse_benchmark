import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_run_inversion import run_inversion

# --- INJECTED REFEREE: evaluate_results ---
def evaluate_results(y_true, y_pred, recon_object):
    """
    Calculate metrics (PSNR) and display results.
    Note: For this specific validation, the 'y_true' in the signature usually refers to the 
    Ground Truth Projection (if doing forward projection) or Ground Truth Object (if doing reconstruction).
    
    However, the QA instruction implies we are comparing 'Agent's quality against Ground Truth'.
    The standard data typically contains inputs (Projection 'y') and output (Reconstructed Object 'result').
    
    In the context of standard evaluation for Inverse Abel:
    - y_true: The input projection (used for reprojection consistency checks or visualization).
    - recon_object: The result of the inversion (the object).
    - y_pred: The re-projection of the reconstructed object (to compare with y_true).
    
    BUT, looking at the function signature in Reference B:
    evaluate_results(y_true, y_pred, recon_object)
    
    And the metric calculation:
    mse = np.mean((y_true - y_pred) ** 2)
    
    This implies y_true and y_pred are comparable (both projections).
    
    Wait, the validation goal is to compare the *Agent's output* vs the *Standard output*?
    No, the instructions say "compare the Agent's quality against the Ground Truth".
    
    Since we only have the standard .pkl which contains (Input Projection, Output Object), 
    and we don't necessarily have a forward projector in this script to generate y_pred from recon_object 
    (unless we write one, or the evaluation function does it, but the injected code *takes* y_pred as an argument).
    
    Let's look at the usage in the instruction:
    "Logic: Load inputs -> Run run_inversion -> Get Result -> Evaluate."
    
    The provided evaluate_results function signature is tricky: `evaluate_results(y_true, y_pred, recon_object)`.
    It calculates MSE between `y_true` and `y_pred`.
    
    If we don't have a forward projector to calculate `y_pred` from `recon_object`, we might need to adapt.
    However, often in these specific QA tasks, if we strictly validate against *Standard Recorded Output*,
    we can compare Agent Result vs Standard Result directly.
    
    BUT, the instructions explicitly ask to use `evaluate_results`.
    
    Let's re-read the provided `evaluate_results` code carefully.
    It takes `y_pred`. It plots `y_pred`.
    If we cannot generate `y_pred` (forward projection of our reconstruction), we cannot validly use this specific `evaluate_results` without mocking `y_pred` or having a forward projector.
    
    However, `run_inversion` is an INVERSE transform.
    The standard data output is the RECONSTRUCTED object.
    
    Let's look at the instruction again:
    "Evaluate: score_agent = evaluate_results(final_result)" -> This conflicts with the signature `(y_true, y_pred, recon_object)`.
    
    Correction Strategy:
    The instruction likely assumes a generic evaluation flow, but provided a specific `evaluate_results` that requires 3 args.
    If I cannot run a forward projection, I will modify the usage to compare the Agent's Result directly against the Standard Result (Ground Truth Object) for the purpose of the MSE calculation, treating:
    - y_true = Standard Result (from pkl)
    - y_pred = Agent Result
    - recon_object = Agent Result (for plotting)
    
    This calculates the PSNR between the *Code's output* and the *Recorded output*, which effectively verifies integrity.
    "Validation of performance integrity" -> Does the new code produce the same result as the old code?
    """
    
    # ADAPTATION for QA Script Validation:
    # If the user passed fewer arguments in the calling block because they don't have a forward projector:
    # We treat y_true as Ground Truth (Standard Output), y_pred as Agent Output.
    
    # MSE and PSNR calculation
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        # Avoid zero division if max is 0 (empty image)
        pixel_max = max(np.max(y_true), np.max(y_pred))
        if pixel_max == 0: pixel_max = 1.0
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(y_true, cmap='viridis')
    plt.title("Standard Result (GT)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon_object, cmap='magma', vmax=np.max(recon_object)*0.5 if np.max(recon_object) > 0 else 1)
    plt.title("Agent Reconstructed Object")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred, cmap='viridis')
    plt.title(f"Agent Result\nPSNR: {psnr:.1f} dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_results.png')
    print("Results saved to reconstruction_results.png")
    
    return psnr
# --- END INJECTED REFEREE ---


def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def main():
    # 1. Configuration
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 2. Logic to handle Outer/Inner execution patterns
    outer_data = None
    inner_data = None
    
    # Simple heuristic to classify files
    for path in data_paths:
        if "parent_function" in path:
            inner_data = load_data(path)
        else:
            outer_data = load_data(path)
            
    if outer_data is None:
        print("Error: No primary outer data found.")
        sys.exit(1)

    print(f"Running test for: {outer_data.get('func_name')}")

    try:
        # 3. Execution Phase
        # Extract inputs for the outer function
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run the Agent's Code
        # This is 'run_inversion'
        print("Executing run_inversion...")
        agent_result = run_inversion(*outer_args, **outer_kwargs)
        
        final_result = agent_result
        std_result = outer_data.get('output')

        # Handle Chained Execution (if inner_data exists and result is callable)
        # Note: run_inversion usually returns a numpy array, not a function.
        # But we implement the logic just in case it's a factory pattern as per instructions.
        if inner_data and callable(agent_result):
            print("Detected callable output. Executing inner function...")
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            final_result = agent_result(*inner_args, **inner_kwargs)
            std_result = inner_data.get('output')
        
        # 4. Evaluation Phase
        # We compare Agent Result vs Standard Recorded Result.
        # Logic: High PSNR between them means the Agent code reproduces the Standard behavior correctly.
        
        # Ensure numpy arrays
        final_result = np.array(final_result)
        std_result = np.array(std_result)
        
        # Handle shape mismatches (squeeze if necessary, though uncommon in Abel transforms)
        if final_result.shape != std_result.shape:
             print(f"Warning: Shape mismatch. Agent: {final_result.shape}, Std: {std_result.shape}")
             # Attempt trivial fix
             if final_result.size == std_result.size:
                 final_result = final_result.reshape(std_result.shape)

        print("\n--- Evaluating Agent Output vs Standard Output ---")
        # In this context:
        # y_true = std_result (The trusted recording)
        # y_pred = final_result (The current agent's output)
        # recon_object = final_result (Visual aid)
        score_psnr = evaluate_results(std_result, final_result, final_result)
        
        # 5. Verification
        # Since we are comparing the code against its own recorded past execution (regression testing),
        # we expect extremely high similarity (near infinity PSNR ideally, or > 50dB depending on float precision).
        # We set a threshold.
        
        THRESHOLD_PSNR = 40.0 # dB. If it's below this, the logic has changed significantly.
        
        print(f"Validation Score (PSNR): {score_psnr:.2f} dB")
        
        if score_psnr < THRESHOLD_PSNR:
            print(f"FAIL: PSNR {score_psnr:.2f} dB is below threshold {THRESHOLD_PSNR} dB.")
            print("Performance integrity check failed.")
            sys.exit(1)
        else:
            print(f"SUCCESS: PSNR {score_psnr:.2f} dB is acceptable.")
            sys.exit(0)

    except Exception as e:
        print(f"An error occurred during execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()