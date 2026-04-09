import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# --- Inject the Referee (Evaluation Logic) ---
def evaluate_results(reconstruction, so2, output_file="pat_result.png"):
    """
    Evaluate and visualize the reconstruction and sO2 results.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        so2: Oxygen saturation map, shape (nz, ny, nx)
        output_file: Path to save the output figure
        
    Returns:
        mean_so2: Mean sO2 value in the ROI
    """
    recon_img = np.mean(reconstruction, axis=0)[0]
    so2_img = so2[0]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(recon_img.T, cmap='gray', origin='lower')
    plt.title("Reconstruction (Mean WL)")
    plt.colorbar(label="PA Signal")
    
    plt.subplot(1, 2, 2)
    plt.imshow(so2_img.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.title("sO2 Estimation")
    plt.colorbar(label="sO2")
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    valid_so2 = so2_img[so2_img > 0]
    if len(valid_so2) > 0:
        mean_so2 = np.mean(valid_so2)
    else:
        mean_so2 = 0.0
    
    print(f"Mean sO2 in ROI: {mean_so2:.4f}")
    
    recon_max = np.max(reconstruction)
    recon_min = np.min(reconstruction)
    print(f"Reconstruction range: [{recon_min:.4f}, {recon_max:.4f}]")
    
    return mean_so2


def evaluate_reconstruction_only(reconstruction, output_file="pat_result.png"):
    """
    Evaluate reconstruction when sO2 is not available.
    Uses reconstruction quality metrics instead.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        output_file: Path to save the output figure
        
    Returns:
        quality_score: A quality metric based on reconstruction
    """
    recon_img = np.mean(reconstruction, axis=0)[0]
    
    plt.figure(figsize=(6, 5))
    plt.imshow(recon_img.T, cmap='gray', origin='lower')
    plt.title("Reconstruction (Mean WL)")
    plt.colorbar(label="PA Signal")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    recon_max = np.max(reconstruction)
    recon_min = np.min(reconstruction)
    recon_mean = np.mean(reconstruction)
    recon_std = np.std(reconstruction)
    
    print(f"Reconstruction range: [{recon_min:.4f}, {recon_max:.4f}]")
    print(f"Reconstruction mean: {recon_mean:.4f}, std: {recon_std:.4f}")
    
    # Return a composite quality score
    # Higher dynamic range and signal strength indicate better reconstruction
    quality_score = recon_max - recon_min + recon_std
    
    return quality_score


def compare_reconstructions(agent_recon, std_recon):
    """
    Compare two reconstructions directly.
    
    Args:
        agent_recon: Agent's reconstruction
        std_recon: Standard reconstruction
        
    Returns:
        similarity_score: A score indicating how similar the reconstructions are (higher is better)
    """
    # Normalize both reconstructions
    agent_norm = (agent_recon - np.min(agent_recon)) / (np.max(agent_recon) - np.min(agent_recon) + 1e-10)
    std_norm = (std_recon - np.min(std_recon)) / (np.max(std_recon) - np.min(std_recon) + 1e-10)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(agent_norm.flatten(), std_norm.flatten())[0, 1]
    
    # Calculate normalized RMSE
    rmse = np.sqrt(np.mean((agent_norm - std_norm) ** 2))
    
    # Calculate structural similarity (simple version)
    mean_agent = np.mean(agent_norm)
    mean_std = np.mean(std_norm)
    std_agent = np.std(agent_norm)
    std_std = np.std(std_norm)
    covariance = np.mean((agent_norm - mean_agent) * (std_norm - mean_std))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mean_agent * mean_std + c1) * (2 * covariance + c2)) / \
           ((mean_agent ** 2 + mean_std ** 2 + c1) * (std_agent ** 2 + std_std ** 2 + c2))
    
    print(f"Correlation: {correlation:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"SSIM: {ssim:.4f}")
    
    # Combined similarity score (higher is better)
    similarity_score = correlation * 0.5 + ssim * 0.5
    
    return similarity_score


def main():
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs...")
        
        # Execute the target function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is a chained execution pattern
        if inner_data_files:
            # Pattern 2: Chained Execution
            print("Detected chained execution pattern...")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Pattern 1: Direct Execution
            print("Detected direct execution pattern...")
            final_result = agent_output
            std_result = std_output
        
        print(f"Agent output type: {type(final_result)}")
        print(f"Standard output type: {type(std_result)}")
        
        # Extract reconstruction from results
        if isinstance(final_result, dict):
            agent_reconstruction = final_result.get('reconstruction', None)
            agent_so2 = final_result.get('so2', None)
        else:
            agent_reconstruction = final_result
            agent_so2 = None
        
        if isinstance(std_result, dict):
            std_reconstruction = std_result.get('reconstruction', None)
            std_so2 = std_result.get('so2', None)
        else:
            std_reconstruction = std_result
            std_so2 = None
        
        print(f"Agent reconstruction shape: {agent_reconstruction.shape if agent_reconstruction is not None else None}")
        print(f"Standard reconstruction shape: {std_reconstruction.shape if std_reconstruction is not None else None}")
        
        # Evaluation Phase
        if agent_so2 is not None and std_so2 is not None:
            # Use the full evaluate_results function
            print("\n--- Evaluating Agent Result ---")
            score_agent = evaluate_results(agent_reconstruction, agent_so2, output_file="agent_result.png")
            
            print("\n--- Evaluating Standard Result ---")
            score_std = evaluate_results(std_reconstruction, std_so2, output_file="std_result.png")
            
            print(f"\nScores -> Agent: {score_agent}, Standard: {score_std}")
            
            # For sO2, we want values to be similar (within tolerance)
            tolerance = 0.1  # 10% tolerance
            if abs(score_agent - score_std) <= tolerance or score_agent >= score_std * 0.9:
                print("PASS: Agent performance is acceptable.")
                sys.exit(0)
            else:
                print("FAIL: Agent performance degraded significantly.")
                sys.exit(1)
        else:
            # Use reconstruction comparison
            print("\n--- Comparing Reconstructions Directly ---")
            
            if agent_reconstruction is not None and std_reconstruction is not None:
                # Evaluate individual reconstructions
                print("\n--- Agent Reconstruction Quality ---")
                score_agent = evaluate_reconstruction_only(agent_reconstruction, output_file="agent_result.png")
                
                print("\n--- Standard Reconstruction Quality ---")
                score_std = evaluate_reconstruction_only(std_reconstruction, output_file="std_result.png")
                
                print(f"\nQuality Scores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
                
                # Compare reconstructions directly
                print("\n--- Direct Comparison ---")
                similarity = compare_reconstructions(agent_reconstruction, std_reconstruction)
                print(f"Similarity Score: {similarity:.4f}")
                
                # Success criteria:
                # 1. Similarity should be high (> 0.8)
                # 2. Quality score should be comparable (within 10%)
                if similarity > 0.8 or score_agent >= score_std * 0.9:
                    print("PASS: Agent performance is acceptable.")
                    sys.exit(0)
                else:
                    print("FAIL: Agent performance degraded significantly.")
                    sys.exit(1)
            else:
                print("ERROR: Could not extract reconstructions for comparison.")
                sys.exit(1)
                
    except Exception as e:
        print(f"ERROR: Exception occurred during testing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()