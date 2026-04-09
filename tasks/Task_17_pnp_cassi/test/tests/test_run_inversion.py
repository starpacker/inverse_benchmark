import sys
import os
import dill
import numpy as np
import traceback
import math
import scipy.io as sio
import matplotlib.pyplot as plt

# Import target function
from agent_run_inversion import run_inversion


# --- Injected Referee (Evaluation Logic) ---

def psnr(ref, img):
    """
    Peak signal-to-noise ratio (PSNR).
    """
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate_results(recon_img, truth, psnrs, output_dir='.'):
    """
    Evaluate and save reconstruction results.
    """
    # Final PSNR
    final_psnr = psnr(truth, recon_img)
    print(f"Final PSNR: {final_psnr:.2f} dB")

    # Save Reconstruction as .mat
    sio.savemat(os.path.join(output_dir, 'recon_result.mat'), {'img': recon_img})

    # Save spectral channels as image grid
    nC = recon_img.shape[2]
    fig = plt.figure(figsize=(10, 10))
    for i in range(min(9, nC)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(recon_img[:, :, i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'Band {i+1}')
    plt.savefig(os.path.join(output_dir, 'recon_channels.png'))
    plt.close()

    # Save PSNR plot
    plt.figure()
    plt.plot(psnrs)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction Convergence')
    plt.savefig(os.path.join(output_dir, 'psnr_curve.png'))
    plt.close()

    return final_psnr


def main():
    # Data paths provided
    data_paths = ['/home/yjh/pnp_cassi_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Outer data keys: {outer_data.keys()}")
        print(f"Number of args: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Execute run_inversion
        print("\n--- Running run_inversion ---")
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Pattern 2: Chained Execution
            print("\n--- Chained Execution Pattern Detected ---")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # agent_output should be a callable (operator)
            print("Executing operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Pattern 1: Direct Execution
            print("\n--- Direct Execution Pattern ---")
            final_result = agent_output
            std_result = std_output
        
        # Extract results - run_inversion returns (x_img, psnr_all)
        if isinstance(final_result, tuple) and len(final_result) == 2:
            agent_recon_img, agent_psnrs = final_result
        else:
            print("ERROR: Unexpected output format from run_inversion")
            sys.exit(1)
        
        if isinstance(std_result, tuple) and len(std_result) == 2:
            std_recon_img, std_psnrs = std_result
        else:
            print("ERROR: Unexpected output format from standard result")
            sys.exit(1)
        
        # Get truth from the input args
        # run_inversion signature: (meas, Phi, truth, _lambda=1, iter_max=20, tv_weight=6, tv_iter_max=5, step=1)
        # truth is the 3rd argument (index 2)
        truth = args[2] if len(args) > 2 else kwargs.get('truth', None)
        
        if truth is None:
            print("WARNING: Truth not found in inputs, using std_recon_img for comparison")
            truth = std_recon_img
        
        # Create output directories
        agent_output_dir = './agent_output'
        std_output_dir = './std_output'
        os.makedirs(agent_output_dir, exist_ok=True)
        os.makedirs(std_output_dir, exist_ok=True)
        
        # Evaluation Phase
        print("\n--- Evaluating Agent Results ---")
        score_agent = evaluate_results(agent_recon_img, truth, agent_psnrs, output_dir=agent_output_dir)
        
        print("\n--- Evaluating Standard Results ---")
        score_std = evaluate_results(std_recon_img, truth, std_psnrs, output_dir=std_output_dir)
        
        # Verification & Reporting
        print(f"\nScores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
        
        # PSNR is "Higher is better"
        # Allow 10% margin of error
        margin = 0.90
        threshold = score_std * margin
        
        print(f"Threshold (90% of standard): {threshold:.4f}")
        
        if score_agent >= threshold:
            print("SUCCESS: Agent performance is acceptable!")
            sys.exit(0)
        else:
            print(f"FAILURE: Agent performance ({score_agent:.4f}) is below threshold ({threshold:.4f})")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Exception occurred during testing")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()