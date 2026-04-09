import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.fft

# Ensure target module is in path
sys.path.append('/data/yjh/tomopy-master_sandbox/run_code')

try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Could not import run_inversion from agent_run_inversion.py")
    sys.exit(1)

# --- INJECTED REFEREE (Evaluation Logic) ---
def calculate_psnr(gt, recon):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return 100
    max_pixel = gt.max()
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(gt, recon):
    """Structural Similarity Index Wrapper"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)
    except ImportError:
        return 0

def norm_minmax(x):
    if x.max() - x.min() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())

def evaluate_results(gt, recon_dict):
    """
    Calculates metrics and generates visualization.
    gt: Ground Truth Image
    recon_dict: Dictionary {method_name: reconstructed_image}
    """
    # Ensure inputs are numpy arrays
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)
    
    gt_norm = norm_minmax(gt)
    
    # Create circular mask
    h, w = gt_norm.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - w/2)**2 + (y - h/2)**2 <= (w/2)**2
    
    results_stats = {}
    results_text = []
    
    for name, recon in recon_dict.items():
        if not isinstance(recon, np.ndarray):
            recon = np.array(recon)
            
        r_norm = norm_minmax(recon)
        
        # Calculate PSNR on masked area
        psnr = calculate_psnr(gt_norm[mask], r_norm[mask])
        ssim = calculate_ssim(gt_norm, r_norm)
        
        results_text.append(f"{name} -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        results_stats[name] = {'psnr': psnr, 'ssim': ssim}
        print(results_text[-1])

    # Visualization
    try:
        num_plots = 1 + len(recon_dict)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1: axes = [axes] # Handle single plot case if empty dict
        elif not isinstance(axes, (list, np.ndarray)): axes = [axes]
        
        # Plot GT
        axes[0].imshow(gt, cmap='gray')
        axes[0].set_title('Ground Truth / Reference')
        axes[0].axis('off')
        
        # Plot Recons
        for i, (name, recon) in enumerate(recon_dict.items(), 1):
            if i < len(axes):
                axes[i].imshow(recon, cmap='gray')
                axes[i].set_title(name)
                axes[i].axis('off')
            
        output_file = 'tomopy_workflow_refactored.png'
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        
    return results_stats

# --- HELPER FUNCTIONS FOR EXECUTION ---
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def get_ground_truth_for_evaluation(input_args, input_kwargs):
    """
    In typical inverse problems, we don't always have the ground truth image 
    passed explicitly into the reconstruction function. 
    However, for evaluation purposes, we need a reference.
    
    Strategy: 
    1. Check if 'phantom' or 'gt' is in kwargs (sometimes stored there for dev).
    2. If not, we will use the Standard Implementation's output as the "Ground Truth" 
       (Regression Testing) to ensure the Agent matches the Standard.
    """
    # Ideally, we would reconstruct a phantom if the inputs allowed, but here 
    # we treat the Standard Output (the saved .pkl output) as the Reference.
    return None

def main():
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 1. Identify File Patterns
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        if 'parent_function' in path:
            inner_data_paths.append(path)
        else:
            outer_data_path = path

    if not outer_data_path:
        print("No primary data file found (standard_data_run_inversion.pkl).")
        sys.exit(1)

    print(f"Loading primary data from: {outer_data_path}")
    outer_data = load_pkl(outer_data_path)
    
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    std_result_outer = outer_data.get('output')

    # 2. Execution Logic
    # Case A: Chained Execution (Factory/Closure)
    if inner_data_paths and callable(std_result_outer):
        print("Pattern identified: Chained Execution (Factory)")
        
        # Step A1: Run Outer Function to get Agent Operator
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"Agent outer function failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Iterate through inner calls
        for inner_path in inner_data_paths:
            print(f"Processing inner data: {inner_path}")
            inner_data = load_pkl(inner_path)
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            std_final_result = inner_data.get('output')

            # Step A2: Run Agent Operator with Inner Data
            try:
                agent_final_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Agent inner operator failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Evaluate
            # Since we likely don't have the original Phantom in the inputs,
            # we compare Agent Result vs Standard Result (Regression Test).
            # The 'Standard Result' acts as the Ground Truth here.
            
            recon_dict = {"Agent_Recon": agent_final_result}
            
            # Using Standard Result as GT for metric calculation
            scores = evaluate_results(std_final_result, recon_dict)
            
            # Validation Check
            psnr = scores['Agent_Recon']['psnr']
            if psnr < 40: # High threshold because it should be nearly identical to standard code
                print(f"FAILURE: Agent result deviates from Standard result. PSNR: {psnr}")
                sys.exit(1)

    # Case B: Direct Execution
    else:
        print("Pattern identified: Direct Execution")
        
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"Agent execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        std_result = std_result_outer

        # Evaluate
        # We compare Agent Result vs Standard Result (Regression Test).
        recon_dict = {
            "Agent_Recon": agent_result
        }
        
        # Check shapes
        if agent_result.shape != std_result.shape:
            print(f"Shape mismatch! Agent: {agent_result.shape}, Std: {std_result.shape}")
            sys.exit(1)

        print("Evaluating Agent performance against Standard Output (Regression)...")
        scores = evaluate_results(std_result, recon_dict)
        
        # Metric extraction
        agent_psnr = scores['Agent_Recon']['psnr']
        
        # Threshold: Since we are comparing against the output of the "Standard" code 
        # (which is likely the same algorithm), the images should be extremely similar.
        # PSNR should be very high (near infinity) if logic is identical, or decent (>30dB) if floating point diffs.
        if agent_psnr < 50: 
            print(f"WARNING: Low PSNR ({agent_psnr:.2f}) when comparing Agent to Standard output.")
            # If strictly identical logic is expected, this might be a fail. 
            # However, allow some wiggle room for env differences.
            if agent_psnr < 25:
                print("FAILURE: Significant deviation from standard output.")
                sys.exit(1)
        else:
            print(f"SUCCESS: Agent matches Standard output with high fidelity (PSNR: {agent_psnr:.2f}).")

    sys.exit(0)

if __name__ == "__main__":
    main()