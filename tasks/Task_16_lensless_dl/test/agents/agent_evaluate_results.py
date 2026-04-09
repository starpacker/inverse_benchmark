import numpy as np
import matplotlib.pyplot as plt

def evaluate_results(reconstruction, output_path="result.png"):
    """
    Evaluate and save reconstruction results.
    
    Args:
        reconstruction: Reconstructed image array
        output_path: Path to save the result image
    """
    print(f"Saving result to {output_path}...")
    
    # Create a copy to avoid modifying the original input
    img = reconstruction.copy()
    
    # Handle batch dimension (e.g., [1, H, W, C] -> [H, W, C])
    if len(img.shape) == 4:
        img = img[0]
    
    # Normalize image for display [0, 1]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_display = (img - img_min) / (img_max - img_min)
    else:
        img_display = img
    
    # Clip to ensure valid range
    img_display = np.clip(img_display, 0, 1)
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Handle different channel configurations for display
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
    
    # Save raw numpy array
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")
    
    # Print statistics
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction min: {reconstruction.min():.6f}")
    print(f"Reconstruction max: {reconstruction.max():.6f}")
    print(f"Reconstruction mean: {reconstruction.mean():.6f}")