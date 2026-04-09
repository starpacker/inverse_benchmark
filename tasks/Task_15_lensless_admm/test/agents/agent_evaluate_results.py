import numpy as np
import matplotlib.pyplot as plt
from lensless.utils.plot import plot_image

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
    # 1. Sanity Check: Print shape and range
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction min: {reconstruction.min():.4f}, max: {reconstruction.max():.4f}")
    
    # 2. Visualization: Plot and save figure
    print(f"Saving result to {output_path}...")
    ax = plot_image(reconstruction, gamma=None)
    
    # Handle potential subplot grid return
    if hasattr(ax, "__len__"):
        ax = ax[0, 0]
        
    ax.set_title("ADMM Reconstruction")
    plt.savefig(output_path)
    plt.close()
    
    # 3. Persistence: Save raw numpy array
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")
    
    # 4. Statistics: Compute and print mean/std
    mean_val = np.mean(reconstruction)
    std_val = np.std(reconstruction)
    print(f"Reconstruction statistics - Mean: {mean_val:.4f}, Std: {std_val:.4f}")