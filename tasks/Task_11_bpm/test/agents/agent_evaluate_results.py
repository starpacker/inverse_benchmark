import numpy as np

def evaluate_results(delta_ri, loss_history, ROI):
    """
    Evaluate and report reconstruction results.
    
    Args:
        delta_ri: Reconstructed refractive index (numpy array)
        loss_history: List of loss values during optimization
        ROI: Region of interest for evaluation (tuple of 6 integers)
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Unpack Region of Interest coordinates
    s0, e0, s1, e1, s2, e2 = ROI
    
    # Extract the sub-volume
    roi_data = delta_ri[s0:e0, s1:e1, s2:e2]
    
    # Compute statistical metrics
    # We cast to float to ensure JSON serializability if needed later
    vmin = float(np.min(roi_data))
    vmax = float(np.max(roi_data))
    vmean = float(np.mean(roi_data))
    vstd = float(np.std(roi_data))
    
    # Handle loss history edge cases (empty list)
    initial_loss = loss_history[0] if len(loss_history) > 0 else None
    final_loss = loss_history[-1] if len(loss_history) > 0 else None
    
    # Aggregate metrics into a dictionary
    metrics = {
        'roi_min': vmin,
        'roi_max': vmax,
        'roi_mean': vmean,
        'roi_std': vstd,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'num_iterations': len(loss_history),
        'loss_history': loss_history
    }
    
    # Print formatted report
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROI Statistics:")
    print(f"  Min: {vmin:.6f}")
    print(f"  Max: {vmax:.6f}")
    print(f"  Mean: {vmean:.6f}")
    print(f"  Std: {vstd:.6f}")
    print(f"Optimization:")
    if initial_loss is not None:
        print(f"  Initial Loss: {initial_loss:.6f}")
    else:
        print("  Initial Loss: N/A")
    if final_loss is not None:
        print(f"  Final Loss: {final_loss:.6f}")
    else:
        print("  Final Loss: N/A")
    print(f"  Iterations: {len(loss_history)}")
    print("=" * 60)
    
    return metrics