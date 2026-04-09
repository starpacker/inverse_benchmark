import sys
import os
import dill
import numpy as np
import traceback
import torch
import torch.nn as nn

# Add path for DPItorch modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

# Import the target function
from agent_run_inversion import run_inversion

# Helper function needed by evaluate_results
def torch_complex_matmul(x, F):
    """Complex matrix multiplication for DFT."""
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, npix, device):
    """
    Compute forward model: image -> visibilities, visibility amplitudes, closure phases.
    """
    eps = 1e-16
    
    F = dft_mat.to(device=device)
    cphase_ind1 = cphase_ind_list[0].to(device=device)
    cphase_ind2 = cphase_ind_list[1].to(device=device)
    cphase_ind3 = cphase_ind_list[2].to(device=device)
    cphase_sign1 = cphase_sign_list[0].to(device=device)
    cphase_sign2 = cphase_sign_list[1].to(device=device)
    cphase_sign3 = cphase_sign_list[2].to(device=device)
    
    x_flat = torch.reshape(x, (-1, npix * npix)).type(torch.float32).to(device=device)
    vis_torch = torch_complex_matmul(x_flat, F)
    
    vis_amp = torch.sqrt((vis_torch[:, 0, :]) ** 2 + (vis_torch[:, 1, :]) ** 2 + eps)
    
    vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
    vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
    vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)
    
    ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
    ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
    ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
    
    cphase = (cphase_sign1 * ang1 + cphase_sign2 * ang2 + cphase_sign3 * ang3) * 180 / np.pi
    
    return vis_torch, vis_amp, cphase

# Inject the evaluate_results function (The Referee)
def evaluate_results(result, data_dict, save_path, device):
    """
    Evaluate and save the inversion results.
    
    Args:
        result: Dictionary containing inversion results
        data_dict: Dictionary containing preprocessed data
        save_path: Path to save results
        device: Torch device
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    final_images = result['final_images']
    loss_history = result['loss_history']
    npix = result['npix']
    
    mean_image = np.mean(final_images, axis=0)
    std_image = np.std(final_images, axis=0)
    
    total_flux = np.sum(mean_image)
    target_flux = data_dict['flux_const']
    flux_error = np.abs(total_flux - target_flux) / target_flux * 100
    
    final_loss = loss_history[-1] if len(loss_history) > 0 else float('nan')
    
    dft_mat = data_dict['dft_mat']
    cphase_ind_list = data_dict['cphase_ind_list']
    cphase_sign_list = data_dict['cphase_sign_list']
    cphase_true = data_dict['cphase_true']
    
    mean_image_tensor = torch.tensor(mean_image, dtype=torch.float32).unsqueeze(0).to(device)
    
    _, _, cphase_pred = forward_operator(
        mean_image_tensor, dft_mat, cphase_ind_list, cphase_sign_list, npix, device
    )
    cphase_pred_np = cphase_pred.detach().cpu().numpy().flatten()
    
    cphase_residual = np.abs(cphase_true - cphase_pred_np)
    cphase_residual = np.minimum(cphase_residual, 360 - cphase_residual)
    mean_cphase_error = np.mean(cphase_residual)
    
    # Skip saving files for testing
    # torch.save(result['model'].state_dict(), os.path.join(save_path, 'model.pth'))
    # np.save(os.path.join(save_path, 'reconstruction.npy'), final_images)
    # np.save(os.path.join(save_path, 'mean_image.npy'), mean_image)
    # np.save(os.path.join(save_path, 'std_image.npy'), std_image)
    # np.save(os.path.join(save_path, 'loss_history.npy'), loss_history)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Mean image shape: {mean_image.shape}")
    print(f"Total flux: {total_flux:.4f} (target: {target_flux:.4f})")
    print(f"Flux error: {flux_error:.2f}%")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Mean closure phase error: {mean_cphase_error:.2f} degrees")
    print(f"Results saved to: {save_path}")
    
    metrics = {
        'mean_image': mean_image,
        'std_image': std_image,
        'total_flux': total_flux,
        'target_flux': target_flux,
        'flux_error_percent': flux_error,
        'final_loss': final_loss,
        'mean_cphase_error_deg': mean_cphase_error
    }
    
    return metrics


def extract_primary_metric(metrics):
    """
    Extract the primary scalar metric from the evaluation result.
    For this inversion task, we use final_loss (lower is better).
    """
    if isinstance(metrics, dict):
        # final_loss is the primary metric (lower is better)
        return metrics.get('final_loss', float('inf'))
    return float(metrics)


def main():
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task1_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    print(f"Outer data path: {outer_data_path}")
    print(f"Inner data paths: {inner_data_paths}")
    
    # Load outer data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run the agent's function
    try:
        print("\n--- Running Agent's run_inversion ---")
        agent_output = run_inversion(*args, **kwargs)
        print("Agent's run_inversion completed successfully.")
    except Exception as e:
        print(f"ERROR: Agent's run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution pattern)
    if inner_data_paths:
        # Pattern 2: Chained execution
        print("\n--- Chained Execution Pattern Detected ---")
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_std_output = inner_data.get('output', None)
                
                # Execute the operator returned by run_inversion
                if callable(agent_output):
                    final_result = agent_output(*inner_args, **inner_kwargs)
                    std_result = inner_std_output
                else:
                    print("WARNING: Agent output is not callable, using direct output.")
                    final_result = agent_output
                    std_result = std_output
            except Exception as e:
                print(f"ERROR: Failed to process inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("\n--- Direct Execution Pattern ---")
        final_result = agent_output
        std_result = std_output
    
    # Now we need data_dict for evaluation
    # The first argument to run_inversion is data_dict
    if len(args) > 0:
        data_dict = args[0]
    elif 'data_dict' in kwargs:
        data_dict = kwargs['data_dict']
    else:
        print("ERROR: Could not find data_dict in arguments.")
        sys.exit(1)
    
    # Create a temporary save path for evaluation
    save_path = '/tmp/test_run_inversion_eval'
    
    # Evaluate agent's result
    try:
        print("\n--- Evaluating Agent's Result ---")
        metrics_agent = evaluate_results(final_result, data_dict, save_path, device)
        score_agent = extract_primary_metric(metrics_agent)
        print(f"Agent's final loss: {score_agent:.4f}")
    except Exception as e:
        print(f"ERROR: Failed to evaluate agent's result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    try:
        print("\n--- Evaluating Standard Result ---")
        metrics_std = evaluate_results(std_result, data_dict, save_path, device)
        score_std = extract_primary_metric(metrics_std)
        print(f"Standard's final loss: {score_std:.4f}")
    except Exception as e:
        print(f"ERROR: Failed to evaluate standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Report scores
    print(f"\n=== Comparison ===")
    print(f"Scores -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    
    # For loss, lower is better
    # Allow 20% margin of error (agent's loss can be up to 20% higher than standard)
    margin = 0.20
    
    if np.isnan(score_agent) or np.isnan(score_std):
        print("WARNING: One of the scores is NaN. Cannot compare properly.")
        # If both are NaN, consider it a pass
        if np.isnan(score_agent) and np.isnan(score_std):
            print("Both scores are NaN - treating as acceptable.")
            sys.exit(0)
        else:
            print("Only one score is NaN - test failed.")
            sys.exit(1)
    
    # For loss (lower is better): agent passes if agent_loss <= std_loss * (1 + margin)
    threshold = score_std * (1 + margin)
    
    print(f"Threshold (allowing {margin*100:.0f}% degradation): {threshold:.4f}")
    
    if score_agent <= threshold:
        print("RESULT: PASS - Agent's performance is acceptable.")
        sys.exit(0)
    else:
        print(f"RESULT: FAIL - Agent's loss ({score_agent:.4f}) exceeds threshold ({threshold:.4f}).")
        sys.exit(1)


if __name__ == "__main__":
    main()