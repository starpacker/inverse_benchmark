import numpy as np

import matplotlib.pyplot as plt

def evaluate_results(initial_image, final_image, strehls):
    """
    Plot and save results.
    """
    plt.figure(figsize=(12, 4))
    
    vmin = -5
    vmax = 0
    
    plt.subplot(1, 3, 1)
    # Use global imshow logic but implemented locally or assume Field object
    # Since Field is defined, we can assume standard matplotlib for array
    plt.imshow(np.log10(initial_image.shaped.real / initial_image.max() + 1e-10), 
               origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
    plt.title("Aberrated Image (Log)")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log10(final_image.shaped.real / final_image.max() + 1e-10), 
               origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
    plt.title("Corrected Image (Log)")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.plot(strehls)
    plt.title("Strehl Ratio Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Strehl Ratio")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("hcipy_standalone_results.png")
    print("Results saved to hcipy_standalone_results.png")
