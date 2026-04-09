import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(image_result):
    """
    Computes statistics and saves the resulting image.
    """
    # Statistics
    mean_val = np.mean(image_result)
    std_val = np.std(image_result)
    min_val = np.min(image_result)
    max_val = np.max(image_result)
    
    print(f"Evaluation Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}")
    
    # Visualization
    output_filename = "oct_reconstruction_refactored.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(image_result, cmap='gray', aspect='auto')
    plt.title('Refactored OCT Structure Reconstruction')
    plt.colorbar(label='Normalized Intensity')
    plt.xlabel('A-Lines')
    plt.ylabel('Depth (Z)')
    plt.savefig(output_filename)
    print(f"Result saved to {output_filename}")
    
    return output_filename
