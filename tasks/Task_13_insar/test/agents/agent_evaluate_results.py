import numpy as np

def evaluate_results(F, preprocessed_data, outname):
    """
    Save results and print statistics.
    
    Parameters
    ----------
    F : ndarray
        Unwrapped phase result.
    preprocessed_data : dict
        Dictionary containing magnitude and other metadata.
    outname : str
        Output filename.
        
    Returns
    -------
    mean_phase : float
        Mean value of the unwrapped phase.
    """
    mag = preprocessed_data['mag']

    min_val = np.min(F)
    max_val = np.max(F)
    mean_val = np.mean(F)
    std_val = np.std(F)

    print(f"Evaluation: Unwrapped phase range [{min_val}, {max_val}]")
    print(f"Evaluation: Mean={mean_val}, Std={std_val}")

    if outname.endswith(".tif"):
        try:
            import rasterio as rio
            height, width = F.shape
            with rio.open(
                outname,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                dtype=F.dtype,
                count=1,
            ) as dst:
                dst.write(F, 1)
            print(f"Saved result to {outname}")
        except ImportError:
            print("rasterio not found, saving as npy instead")
            np.save(outname.replace(".tif", ".npy"), F)
            print(f"Saved numpy result to {outname.replace('.tif', '.npy')}")

    elif outname.endswith(".unw"):
        unw_with_mag = np.hstack((mag, F))
        unw_with_mag.tofile(outname)
        print(f"Saved binary result to {outname}")
    else:
        # Default fallback, just save npy
        np.save(outname + ".npy", F)
        print(f"Saved numpy result to {outname}.npy")

    return mean_val