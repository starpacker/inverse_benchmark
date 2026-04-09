import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Referee Function ---
def evaluate_results(reconstruction, output_path="result.png"):
    """
    Evaluate and save reconstruction results.
    
    Args:
        reconstruction: Reconstructed image array
        output_path: Path to save the result image
    """
    print(f"Saving result to {output_path}...")
    
    img = reconstruction.copy()
    
    if len(img.shape) == 4:
        img = img[0]
    
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_display = (img - img_min) / (img_max - img_min)
    else:
        img_display = img
    
    img_display = np.clip(img_display, 0, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    if len(img_display.shape) == 3 and img_display.shape[-1] == 3:
        ax.imshow(img_display)
    elif len(img_display.shape) == 3 and img_display.shape[-1] == 1:
        ax.imshow(img_display[:, :, 0], cmap='gray')
    elif len(img_display.shape) == 2:
        ax.imshow(img_display, cmap='gray')
    else:
        ax.imshow(img_display)
    
    ax.set_title("APGD Reconstruction")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")
    
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction min: {reconstruction.min():.6f}")
    print(f"Reconstruction max: {reconstruction.max():.6f}")
    print(f"Reconstruction mean: {reconstruction.mean():.6f}")


def compute_metrics(reconstruction):
    """
    Compute numerical metrics for reconstruction quality evaluation.
    Returns a dictionary of metrics.
    """
    img = reconstruction.copy()
    
    if len(img.shape) == 4:
        img = img[0]
    
    metrics = {
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "l2_norm": float(np.linalg.norm(img)),
        "shape": img.shape
    }
    
    return metrics


def compare_reconstructions(agent_result, std_result):
    """
    Compare agent and standard reconstruction results.
    Returns a score indicating quality (higher is better).
    """
    agent_img = agent_result.copy()
    std_img = std_result.copy()
    
    # Handle 4D arrays
    if len(agent_img.shape) == 4:
        agent_img = agent_img[0]
    if len(std_img.shape) == 4:
        std_img = std_img[0]
    
    # Ensure same shape for comparison
    if agent_img.shape != std_img.shape:
        print(f"Warning: Shape mismatch - Agent: {agent_img.shape}, Standard: {std_img.shape}")
        # Try to compare what we can
        min_shape = tuple(min(a, s) for a, s in zip(agent_img.shape, std_img.shape))
        if len(min_shape) == 3:
            agent_img = agent_img[:min_shape[0], :min_shape[1], :min_shape[2]]
            std_img = std_img[:min_shape[0], :min_shape[1], :min_shape[2]]
        elif len(min_shape) == 2:
            agent_img = agent_img[:min_shape[0], :min_shape[1]]
            std_img = std_img[:min_shape[0], :min_shape[1]]
    
    # Normalize both for fair comparison
    agent_norm = (agent_img - agent_img.min()) / (agent_img.max() - agent_img.min() + 1e-10)
    std_norm = (std_img - std_img.min()) / (std_img.max() - std_img.min() + 1e-10)
    
    # Compute MSE (lower is better)
    mse = np.mean((agent_norm - std_norm) ** 2)
    
    # Compute PSNR (higher is better)
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = 100.0  # Perfect match
    
    # Compute correlation (higher is better, max 1.0)
    agent_flat = agent_norm.flatten()
    std_flat = std_norm.flatten()
    correlation = np.corrcoef(agent_flat, std_flat)[0, 1]
    
    print(f"Comparison Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Correlation: {correlation:.6f}")
    
    return {
        "mse": mse,
        "psnr": psnr,
        "correlation": correlation
    }


def main():
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 60)
    print("Testing run_inversion Performance")
    print("=" * 60)
    
    # Analyze data paths
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        if 'parent_function' in os.path.basename(path):
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"\nData paths analysis:")
    print(f"  Outer data: {outer_data_path}")
    print(f"  Inner data: {inner_data_paths}")
    
    is_chained = len(inner_data_paths) > 0
    print(f"  Execution pattern: {'Chained' if is_chained else 'Direct'}")
    
    try:
        # Load outer data
        print(f"\nLoading outer data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {list(outer_data.keys())}")
        
        # Extract args and kwargs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Run the agent function
        print("\n" + "-" * 40)
        print("Running agent's run_inversion...")
        print("-" * 40)
        
        agent_output = run_inversion(*args, **kwargs)
        
        print("\nAgent output obtained.")
        
        if is_chained:
            # Chained execution - agent_output is a callable
            print("\nChained execution detected.")
            
            # Load inner data
            inner_data_path = inner_data_paths[0]
            print(f"Loading inner data from: {inner_data_path}")
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print("Running inner function...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        # Evaluate results
        print("\n" + "-" * 40)
        print("Evaluating Results")
        print("-" * 40)
        
        # Save and display agent results
        print("\n--- Agent Reconstruction ---")
        evaluate_results(final_result, output_path="agent_result.png")
        agent_metrics = compute_metrics(final_result)
        print(f"Agent metrics: {agent_metrics}")
        
        # Save and display standard results
        if std_result is not None:
            print("\n--- Standard Reconstruction ---")
            evaluate_results(std_result, output_path="standard_result.png")
            std_metrics = compute_metrics(std_result)
            print(f"Standard metrics: {std_metrics}")
            
            # Compare results
            print("\n--- Comparison ---")
            comparison = compare_reconstructions(final_result, std_result)
            
            # Determine success based on comparison
            # For reconstruction, we expect high correlation and reasonable PSNR
            psnr = comparison["psnr"]
            correlation = comparison["correlation"]
            
            print(f"\nScores -> Agent PSNR: {psnr:.2f} dB, Correlation: {correlation:.4f}")
            
            # Success criteria:
            # - PSNR should be reasonably high (>20 dB for similar reconstructions)
            # - Correlation should be high (>0.8 for similar results)
            # We allow some margin as optimization algorithms may have slight variations
            
            success = True
            if np.isnan(correlation):
                print("Warning: Correlation is NaN, checking other metrics...")
                # Fall back to comparing norms
                agent_norm = agent_metrics["l2_norm"]
                std_norm = std_metrics["l2_norm"]
                norm_ratio = agent_norm / (std_norm + 1e-10)
                print(f"Norm ratio (agent/std): {norm_ratio:.4f}")
                if norm_ratio < 0.1 or norm_ratio > 10:
                    success = False
                    print("FAIL: Norm ratio indicates significantly different results")
            elif correlation < 0.7:
                success = False
                print("FAIL: Correlation too low (< 0.7)")
            
            if psnr < 15 and not np.isinf(psnr):
                # Low PSNR might still be acceptable if correlation is very high
                if correlation < 0.9:
                    success = False
                    print("FAIL: PSNR too low (< 15 dB) with moderate correlation")
            
            if success:
                print("\n" + "=" * 60)
                print("SUCCESS: Agent reconstruction quality is acceptable")
                print("=" * 60)
                sys.exit(0)
            else:
                print("\n" + "=" * 60)
                print("FAILURE: Agent reconstruction quality degraded significantly")
                print("=" * 60)
                sys.exit(1)
        else:
            # No standard output to compare - just verify agent produces valid output
            print("\nNo standard output available for comparison.")
            print("Verifying agent output is valid...")
            
            if final_result is not None and isinstance(final_result, np.ndarray):
                if final_result.size > 0 and not np.all(np.isnan(final_result)):
                    print("SUCCESS: Agent produced valid output")
                    sys.exit(0)
                else:
                    print("FAILURE: Agent output is empty or all NaN")
                    sys.exit(1)
            else:
                print("FAILURE: Agent output is None or not a numpy array")
                sys.exit(1)
                
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Exception during testing")
        print("=" * 60)
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()