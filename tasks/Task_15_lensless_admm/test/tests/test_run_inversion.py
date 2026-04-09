import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# --- Dependencies for evaluate_results ---
# We need to handle the case where lensless might not be available
try:
    from lensless.utils.plot import plot_image
except ImportError:
    # Fallback plot_image function if lensless is not available
    def plot_image(img, gamma=None):
        fig, ax = plt.subplots(1, 1)
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
            ax.imshow(np.clip(img, 0, 1))
        else:
            ax.imshow(img, cmap='gray')
        return np.array([[ax]])


# --- Inject Referee: evaluate_results ---
def evaluate_results(reconstruction, output_path="result.png"):
    """
    Evaluate and save the reconstruction result.
    
    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed image.
    output_path : str
        Path to save the output image.
    """
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction min: {reconstruction.min():.4f}, max: {reconstruction.max():.4f}")
    
    print(f"Saving result to {output_path}...")
    ax = plot_image(reconstruction, gamma=None)
    if hasattr(ax, "__len__"):
        ax = ax[0, 0]
    ax.set_title("ADMM Reconstruction")
    plt.savefig(output_path)
    plt.close()
    
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")
    
    # Compute some basic metrics
    mean_val = np.mean(reconstruction)
    std_val = np.std(reconstruction)
    print(f"Reconstruction statistics - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    
    # Return metrics for comparison
    return {
        'mean': mean_val,
        'std': std_val,
        'min': float(np.min(reconstruction)),
        'max': float(np.max(reconstruction)),
        'shape': reconstruction.shape
    }


def compute_similarity_score(result1, result2):
    """
    Compute a similarity score between two reconstruction results.
    Higher is better.
    """
    # Normalize both results to [0, 1] range for fair comparison
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-8:
            return (x - x_min) / (x_max - x_min)
        return x - x_min
    
    r1_norm = normalize(result1.astype(np.float64))
    r2_norm = normalize(result2.astype(np.float64))
    
    # Compute MSE
    mse = np.mean((r1_norm - r2_norm) ** 2)
    
    # Compute PSNR (use 1.0 as max since normalized)
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = 100.0  # Perfect match
    
    # Compute correlation
    r1_flat = r1_norm.flatten()
    r2_flat = r2_norm.flatten()
    correlation = np.corrcoef(r1_flat, r2_flat)[0, 1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': correlation
    }


def main():
    # Data paths provided
    data_paths = ['/home/yjh/lensless_admm_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    print(f"Outer data file: {outer_data_path}")
    print(f"Inner data files: {inner_data_paths}")
    
    try:
        # Load outer data
        print("\nLoading outer data...")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {kwargs.keys()}")
        
        # Run the agent's implementation
        print("\nRunning agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output type: {type(agent_output)}")
        
        # Check if this is a chained execution pattern
        if inner_data_paths:
            # Chained execution - agent_output should be a callable
            print("\nDetected chained execution pattern...")
            
            for inner_path in inner_data_paths:
                print(f"\nLoading inner data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Execute the operator
                print("Executing operator with inner data...")
                final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("\nDirect execution pattern...")
            final_result = agent_output
            std_result = std_output
        
        # Evaluate both results
        print("\n" + "="*60)
        print("EVALUATION PHASE")
        print("="*60)
        
        print("\n--- Agent Result ---")
        agent_metrics = evaluate_results(final_result, output_path="agent_result.png")
        
        print("\n--- Standard Result ---")
        std_metrics = evaluate_results(std_result, output_path="standard_result.png")
        
        # Compute similarity between results
        print("\n--- Similarity Analysis ---")
        similarity = compute_similarity_score(final_result, std_result)
        print(f"MSE: {similarity['mse']:.6f}")
        print(f"PSNR: {similarity['psnr']:.2f} dB")
        print(f"Correlation: {similarity['correlation']:.4f}")
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Scores -> Agent Mean: {agent_metrics['mean']:.4f}, Standard Mean: {std_metrics['mean']:.4f}")
        print(f"Scores -> Agent Std: {agent_metrics['std']:.4f}, Standard Std: {std_metrics['std']:.4f}")
        print(f"Shape Match: {agent_metrics['shape'] == std_metrics['shape']}")
        
        # Determine success criteria
        # For reconstruction, we want:
        # 1. High correlation (close to 1.0)
        # 2. Reasonable PSNR (> 20 dB is good, > 30 dB is excellent)
        # 3. Similar statistics
        
        success = True
        reasons = []
        
        # Check correlation
        if similarity['correlation'] < 0.9:
            success = False
            reasons.append(f"Low correlation: {similarity['correlation']:.4f} (expected >= 0.9)")
        
        # Check PSNR
        if similarity['psnr'] < 20:
            success = False
            reasons.append(f"Low PSNR: {similarity['psnr']:.2f} dB (expected >= 20 dB)")
        
        # Check shape match
        if agent_metrics['shape'] != std_metrics['shape']:
            success = False
            reasons.append(f"Shape mismatch: {agent_metrics['shape']} vs {std_metrics['shape']}")
        
        # Check if statistics are in reasonable range (within 50% for mean, within factor of 2 for std)
        mean_ratio = agent_metrics['mean'] / (std_metrics['mean'] + 1e-10)
        if mean_ratio < 0.5 or mean_ratio > 2.0:
            # This is a warning, not necessarily a failure
            print(f"WARNING: Mean ratio is {mean_ratio:.2f} (outside 0.5-2.0 range)")
        
        print("\n" + "="*60)
        if success:
            print("TEST PASSED: Agent implementation produces acceptable results")
            print(f"  - Correlation: {similarity['correlation']:.4f}")
            print(f"  - PSNR: {similarity['psnr']:.2f} dB")
            sys.exit(0)
        else:
            print("TEST FAILED: Agent implementation shows significant deviation")
            for reason in reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()