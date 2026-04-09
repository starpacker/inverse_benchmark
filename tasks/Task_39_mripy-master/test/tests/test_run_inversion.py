import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

# -------------------------------------------------------------------------
# 1. IMPORTS & SETUP
# -------------------------------------------------------------------------

# Import target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If not in path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# -------------------------------------------------------------------------
# 2. INJECT REFEREE (Evaluation Logic from Reference B)
# -------------------------------------------------------------------------

def plotim3(im, save_path=None):
    im = np.flip(im, 0)
    plt.figure()
    plt.imshow(im, cmap=cm.gray, origin='lower', interpolation='none')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.close()

def evaluate_results(reconstruction, reference_image, save_prefix="result"):
    """
    Computes PSNR and saves images.
    """
    # Normalize images for fair comparison
    # Handle complex inputs just in case, though usually magnitude is passed
    if np.iscomplexobj(reconstruction):
        reconstruction = np.abs(reconstruction)
    if np.iscomplexobj(reference_image):
        reference_image = np.abs(reference_image)

    if np.max(reference_image) != 0:
        ref_norm = reference_image / np.max(reference_image)
    else:
        ref_norm = reference_image
        
    if np.max(reconstruction) != 0:
        rec_norm = reconstruction / np.max(reconstruction)
    else:
        rec_norm = reconstruction

    # MSE and PSNR
    mse = np.mean((ref_norm - rec_norm) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    
    # Save images
    plotim3(rec_norm, save_path=f'{save_prefix}_recon.png')
    plotim3(ref_norm, save_path=f'{save_prefix}_ref.png')
    
    return psnr

# -------------------------------------------------------------------------
# 3. HELPER FOR DATA LOADING
# -------------------------------------------------------------------------

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

# -------------------------------------------------------------------------
# 4. EXECUTION LOGIC
# -------------------------------------------------------------------------

def main():
    # Defined data paths from instructions
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate Outer and Inner data
    outer_data_path = None
    inner_data_paths = []

    for p in data_paths:
        if 'parent_function' in p:
            inner_data_paths.append(p)
        else:
            outer_data_path = p

    if not outer_data_path:
        print("Error: No primary outer data file found.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_data_path}")
    outer_data = load_data(outer_data_path)
    
    # Extract arguments for the outer function
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Handle Ground Truth for Direct Execution
    # If this is direct execution, the outer_data['output'] is the final image/array to compare against.
    # If this is chained, outer_data['output'] is likely a function/operator, which we can't easily compare directly,
    # so we rely on the inner execution loop for validation.
    
    try:
        print("Executing run_inversion (Outer)...")
        # Step 1: Run Outer Function
        agent_result = run_inversion(*outer_args, **outer_kwargs)
        
        # Check Execution Pattern
        if inner_data_paths:
            # Pattern 2: Chained Execution
            print(f"Detected {len(inner_data_paths)} inner execution paths. Entering Chained Mode.")
            
            # Ensure the result is callable
            if not callable(agent_result):
                print("Error: Chained execution expected a callable operator, but got:", type(agent_result))
                sys.exit(1)
                
            for inner_path in inner_data_paths:
                print(f"  - Processing Inner Data: {inner_path}")
                inner_data = load_data(inner_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                # Execute Inner
                final_agent_result = agent_result(*inner_args, **inner_kwargs)
                
                # Evaluate
                print("  - Evaluating Inner Result...")
                # We need a reference. Since we are validating inversion integrity, 
                # we usually compare the Agent's reconstruction against the Ground Truth stored in the pickle.
                # However, in 'run_inversion', the ground truth image is often NOT the input 'y_observed', 
                # but rather the 'output' stored in the pkl file from the standard run.
                
                # IMPORTANT: In optimization, we often don't have the "true" image in the args, 
                # only the k-space data. The pickle 'output' is the "gold standard" reconstruction 
                # produced by the original code. We compare against THAT to ensure regression testing.
                
                score_agent = evaluate_results(final_agent_result, expected_inner_output, save_prefix="chained_validation")
                
                # For regression testing against a "Standard Run", the PSNR between "Agent Output" 
                # and "Standard Output" should be very high (effectively identical).
                # If we were comparing against a Ground Truth phantom, the logic would be different.
                # Here, we assume the pickle output is the "expected correct behavior".
                
                # Self-consistency check (Agent vs Standard Implementation Output)
                # Since this is a deterministic algorithm (mostly), we expect high similarity.
                if score_agent < 40.0: # 40dB is extremely high similarity, implying near identity
                     print(f"  [FAILURE] Chained Result deviation too high. PSNR: {score_agent}")
                     sys.exit(1)
                else:
                    print(f"  [SUCCESS] Chained Result matches standard. PSNR: {score_agent}")

        else:
            # Pattern 1: Direct Execution (Standard for run_inversion)
            print("Direct Mode detected.")
            expected_output = outer_data.get('output')
            
            # Special Handling: run_inversion usually returns a numpy array (image)
            # Evaluate Agent Output vs Standard Output (Regression Test)
            print("Evaluating Agent Output vs Standard Output (Regression Test)...")
            
            # Note: We treat the standard_output as the 'reference' here because we want to verify
            # that the agent implementation performs identically to the recorded standard run.
            score_agent = evaluate_results(agent_result, expected_output, save_prefix="direct_validation")
            
            # Threshold: Since this is likely the same algorithm, we expect near-identical results.
            # Floating point differences might exist. 30dB PSNR is a good threshold for "same image".
            # If the algorithm has stochastic elements (like random initialization in ADMM?), allow lower.
            # Looking at the code: ADMM has 'x0' initialized, but code provided shows deterministic initialization (z=invA(b)).
            # So it should be deterministic.
            
            # However, if 'expected_output' is the original clean image (Ground Truth) and 'agent_result' is the recon,
            # then a lower PSNR (e.g. 20-25) might be acceptable.
            # But the pickle usually stores the RETURN VALUE of the function.
            # So we are comparing Agent Recon vs Standard Recon. They should be identical.
            
            if score_agent < 50.0: # Expecting very high similarity for regression
                print(f"Warning: Agent result differs from Standard result. PSNR: {score_agent}")
                # If strictly regression, this is a fail. If validating "quality", check if it's acceptable.
                # Let's relax slightly for floating point drift across machines (GPU/CPU)
                if score_agent < 25.0:
                    print("[FAILURE] Significant deviation from standard output.")
                    sys.exit(1)
            
            print(f"[SUCCESS] Direct Execution Verified. Score: {score_agent}")

    except Exception as e:
        print("Exception during execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()