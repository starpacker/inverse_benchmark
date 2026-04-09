import sys
import os
import dill
import numpy as np
import traceback
from scipy.ndimage import shift, gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.registration import phase_cross_correlation

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If the agent file is not in the path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# ===================================================================================
# INJECTED REFEREE: evaluate_results
# ===================================================================================
def evaluate_results(result, data_container):
    """
    Compares reconstruction to ground truth if available and prints metrics.
    
    Args:
        result (dict): Output from run_inversion.
        data_container (dict): Original data dictionary.
        
    Returns:
        tuple: (PSNR, SSIM)
    """
    recon_obj = result['reconstructed_object']
    gt_obj = data_container.get('ground_truth_object')
    positions = data_container['positions']
    No = data_container['No']
    Np = data_container['Np']
    
    if gt_obj is None:
        print("No Ground Truth available for evaluation.")
        return 0.0, 0.0
    
    recon_amp = np.abs(recon_obj)
    gt_amp = np.abs(gt_obj)
    
    print("Evaluating results...")
    
    # 1. Registration (Translation correction)
    # Using subpixel registration on amplitude
    shift_vector, error, diffphase = phase_cross_correlation(gt_amp, recon_amp, upsample_factor=10)
    print(f"  Detected shift: {shift_vector}")
    
    # Apply shift to the complex object
    recon_aligned = shift(recon_obj, shift_vector, mode='wrap')
    recon_amp_aligned = np.abs(recon_aligned)
    
    # 2. ROI Selection (Focus on illuminated area to avoid background noise bias)
    min_r, min_c = np.min(positions, axis=0)
    max_r, max_c = np.max(positions, axis=0)
    roi_slice = (
        slice(max(0, min_r), min(No, max_r + Np)),
        slice(max(0, min_c), min(No, max_c + Np))
    )
    
    recon_roi = recon_amp_aligned[roi_slice]
    gt_roi = gt_amp[roi_slice]
    
    # 3. Scale Matching (Linear Regression y = ax)
    # Match reconstruction intensity to GT intensity
    numerator = np.sum(recon_roi * gt_roi)
    denominator = np.sum(recon_roi**2)
    if denominator < 1e-10: denominator = 1e-10
    scale_opt = numerator / denominator
    
    recon_roi_scaled = recon_roi * scale_opt
    
    # 4. Normalize for Metrics
    max_val = np.max(gt_roi)
    if max_val < 1e-10: max_val = 1.0
    
    recon_final = np.clip(recon_roi_scaled, 0, max_val) / max_val
    gt_final = gt_roi / max_val
    
    # 5. Compute Metrics
    p_val = psnr(gt_final, recon_final, data_range=1.0)
    s_val = ssim(gt_final, recon_final, data_range=1.0)
    
    print(f"  Optimal Scale Factor: {scale_opt:.4f}")
    print(f"  PSNR: {p_val:.2f} dB")
    print(f"  SSIM: {s_val:.4f}")
    
    return p_val, s_val

# ===================================================================================
# TEST LOGIC
# ===================================================================================

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # Defined data paths
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner files
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

    print(f"Loading Outer Data: {outer_file}")
    outer_data = load_pkl(outer_file)
    
    try:
        # --- Step 1: Execute Primary Function ---
        print("Running 'run_inversion' with Outer Data...")
        
        # Extract arguments
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run agent function
        agent_output = run_inversion(*outer_args, **outer_kwargs)
        
        # --- Step 2: Handle Execution Patterns ---
        
        final_result = None
        std_result = None
        data_container = None
        
        if not inner_files:
            # Pattern 1: Direct Execution
            print("Mode: Direct Execution")
            final_result = agent_output
            std_result = outer_data['output']
            
            # The first argument to run_inversion is data_container
            if len(outer_args) > 0:
                data_container = outer_args[0]
            else:
                data_container = outer_kwargs.get('data_container')
                
        else:
            # Pattern 2: Chained Execution (Factory/Closure)
            # Assuming agent_output is a callable
            print(f"Mode: Chained Execution (Found {len(inner_files)} inner files)")
            
            # For simplicity in this template, we take the first inner file to validate logic
            inner_path = inner_files[0] 
            print(f"Processing Inner Data: {inner_path}")
            inner_data = load_pkl(inner_path)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            if not callable(agent_output):
                print("Error: Agent output is not callable, but inner data exists implying a factory pattern.")
                sys.exit(1)
                
            final_result = agent_output(*inner_args, **inner_kwargs)
            std_result = inner_data['output']
            
            # Attempt to locate data_container for evaluation context.
            # In closure patterns, data often resides in the closure or needs to be inferred.
            # Here we try to use arguments from the inner call or fall back to outer.
            if len(inner_args) > 0 and isinstance(inner_args[0], dict):
                 data_container = inner_args[0]
            elif len(outer_args) > 0 and isinstance(outer_args[0], dict):
                 data_container = outer_args[0]

        if data_container is None:
             print("Warning: Could not isolate 'data_container' for evaluation context. Evaluation might fail.")

        # --- Step 3: Evaluation ---
        print("\n--- Evaluating Agent Performance ---")
        agent_psnr, agent_ssim = evaluate_results(final_result, data_container)
        
        # We can also evaluate the standard result to establish a baseline if needed,
        # but the prompt implies comparing Agent Quality against Ground Truth.
        # Often 'std_result' IS the ground truth simulation or a previous good run.
        # However, 'evaluate_results' specifically looks for 'ground_truth_object' inside 'data_container'.
        
        # Let's print the score
        print(f"Agent Scores -> PSNR: {agent_psnr:.2f}, SSIM: {agent_ssim:.4f}")

        # --- Step 4: Success Criteria ---
        # PSNR higher is better.
        # Threshold: We expect decent reconstruction. 
        # A typical starting reconstruction might have PSNR > 15-20dB.
        # Since we don't know the exact difficulty, we set a sanity check > 0.
        # If Ground Truth is missing, evaluate_results returns 0.0.
        
        # In a real pipeline, we might compare agent_psnr vs std_psnr (if std_result was evaluated).
        # Let's try to evaluate std_result just in case to get a relative baseline.
        print("\n--- Evaluating Standard (Recorded) Performance ---")
        std_psnr, std_ssim = evaluate_results(std_result, data_container)
        print(f"Std Scores   -> PSNR: {std_psnr:.2f}, SSIM: {std_ssim:.4f}")
        
        # Compare
        # Allow 10% degradation margin (or absolute drop)
        margin = 0.90
        
        # Check primary metric (PSNR)
        if agent_psnr < (std_psnr * margin) and std_psnr > 1.0: # Only fail if std_psnr is significant
            print(f"FAILURE: Agent PSNR ({agent_psnr:.2f}) is significantly lower than Standard ({std_psnr:.2f})")
            sys.exit(1)
            
        print("SUCCESS: Performance metrics within acceptable range.")
        sys.exit(0)

    except Exception as e:
        print(f"An error occurred during testing:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()