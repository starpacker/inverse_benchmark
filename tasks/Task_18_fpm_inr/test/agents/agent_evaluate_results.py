import os
import numpy as np
import torch

def evaluate_results(result, vis_dir, sample, color):
    """
    Evaluate and save the reconstruction results.
    
    Args:
        result (dict): Dictionary containing 'amplitude', 'phase', 'model', 
                       'final_loss', and 'final_psnr'.
        vis_dir (str): Directory to save visualization .npy files.
        sample (str): Name of the sample (for file naming).
        color (str): Color channel (for file naming).
        
    Returns:
        dict: A dictionary containing statistical metrics of the reconstruction.
    """
    # 1. Unpack the results
    amplitude = result['amplitude']
    phase = result['phase']
    model = result['model']
    final_loss = result['final_loss']
    final_psnr = result['final_psnr']
    
    # 2. Compute statistics for Amplitude
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_min = np.min(amplitude)
    amp_max = np.max(amplitude)
    
    # 3. Compute statistics for Phase
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    
    # 4. Aggregate Metrics
    metrics = {
        'final_loss': final_loss,
        'final_psnr': final_psnr,
        'amplitude_mean': amp_mean,
        'amplitude_std': amp_std,
        'amplitude_min': amp_min,
        'amplitude_max': amp_max,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'phase_min': phase_min,
        'phase_max': phase_max,
    }
    
    # 5. Console Output
    print("\n=== Reconstruction Results ===")
    print(f"Final Loss: {final_loss:.6e}")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Amplitude - Mean: {amp_mean:.4f}, Std: {amp_std:.4f}, Range: [{amp_min:.4f}, {amp_max:.4f}]")
    print(f"Phase - Mean: {phase_mean:.4f}, Std: {phase_std:.4f}, Range: [{phase_min:.4f}, {phase_max:.4f}]")
    
    # 6. Save Model Weights
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{sample}_{color}.pth')
    
    tensors_to_save = []
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    torch.save(tensors_to_save, save_path)
    print(f"Model saved to {save_path}")
    
    # 7. Save Final Images (Raw Data)
    # Ensure vis_dir exists (though usually handled by caller, good practice to check or assume existence)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)
        
    np.save(os.path.join(vis_dir, 'final_amplitude.npy'), amplitude)
    np.save(os.path.join(vis_dir, 'final_phase.npy'), phase)
    print(f"Results saved to {vis_dir}")
    
    return metrics