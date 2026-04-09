import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def evaluate_results(inversion_results, psf_ref, output_path='sh_explicit_results.png'):
    """
    Evaluate and visualize the AO correction results.
    
    Args:
        inversion_results: dict from run_inversion
        psf_ref: reference diffraction-limited PSF
        output_path: path to save the figure
    
    Returns:
        dict: Summary statistics
    """
    strehl_history = inversion_results['strehl_history']
    final_psf = inversion_results['final_psf']
    final_dm_commands = inversion_results['final_dm_commands']
    
    # Compute statistics
    initial_strehl = strehl_history[0] if len(strehl_history) > 0 else 0.0
    final_strehl = strehl_history[-1] if len(strehl_history) > 0 else 0.0
    mean_strehl = np.mean(strehl_history)
    max_strehl = np.max(strehl_history)
    min_strehl = np.min(strehl_history)
    
    # RMS of DM commands
    dm_rms = np.sqrt(np.mean(final_dm_commands ** 2))
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Strehl history
    axes[0].plot(np.arange(1, len(strehl_history) + 1), strehl_history, 'o-', linewidth=2, markersize=4)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Strehl Ratio [%]')
    axes[0].set_title('Strehl Ratio Evolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(strehl_history) + 1])
    
    # Plot 2: Final PSF (log scale)
    psf_display = np.log10(final_psf + 1e-10)
    im1 = axes[1].imshow(psf_display, cmap='hot', origin='lower')
    axes[1].set_title(f'Final PSF (log scale)\nStrehl = {final_strehl:.2f}%')
    axes[1].set_xlabel('Pixels')
    axes[1].set_ylabel('Pixels')
    plt.colorbar(im1, ax=axes[1], label='log10(Intensity)')
    
    # Plot 3: Reference PSF (log scale)
    psf_ref_display = np.log10(psf_ref + 1e-10)
    im2 = axes[2].imshow(psf_ref_display, cmap='hot', origin='lower')
    axes[2].set_title('Reference PSF (Diffraction Limited)')
    axes[2].set_xlabel('Pixels')
    axes[2].set_ylabel('Pixels')
    plt.colorbar(im2, ax=axes[2], label='log10(Intensity)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print summary
    print("\n" + "=" * 60)
    print("                    EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Initial Strehl:     {initial_strehl:.2f}%")
    print(f"  Final Strehl:       {final_strehl:.2f}%")
    print(f"  Mean Strehl:        {mean_strehl:.2f}%")
    print(f"  Max Strehl:         {max_strehl:.2f}%")
    print(f"  Min Strehl:         {min_strehl:.2f}%")
    print(f"  DM RMS Command:     {dm_rms:.2e} m")
    print(f"  Number of Iters:    {len(strehl_history)}")
    print(f"  Output saved to:    {output_path}")
    print("=" * 60)
    
    return {
        'initial_strehl': initial_strehl,
        'final_strehl': final_strehl,
        'mean_strehl': mean_strehl,
        'max_strehl': max_strehl,
        'min_strehl': min_strehl,
        'dm_rms': dm_rms,
        'n_iterations': len(strehl_history)
    }
