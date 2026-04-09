import numpy as np

from scipy.ndimage import rotate as ndrotate

from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")

def run_inversion(cube, angles, n_components=5):
    """
    PCA-based ADI post-processing (inverse solver).
    
    This function performs PCA-ADI reduction:
    1. Reshape cube → (n_frames, n_pixels)
    2. PCA models the quasi-static component (star + halo + speckle)
    3. Subtract PCA model from each frame → residuals contain planet + noise
    4. De-rotate each residual frame by −parallactic angle
    5. Mean combine → planet coherently accumulates, noise averages down
    
    Using mean (not median) because the planet only appears at each pixel
    position in a small fraction of frames; median would remove it.
    
    Parameters
    ----------
    cube : ndarray (n_frames, image_size, image_size)
        ADI observation cube
    angles : ndarray (n_frames,)
        Parallactic angles in degrees
    n_components : int
        Number of PCA components to use for PSF modeling
    
    Returns
    -------
    result : dict containing:
        - 'final_image': (image_size, image_size) mean-combined residual
        - 'derotated': (n_frames, image_size, image_size) derotated residuals
        - 'residuals': (n_frames, image_size, image_size) residuals before derotation
        - 'pca_model': (n_frames, image_size, image_size) PCA-reconstructed cube
    """
    n_frames, ny, nx = cube.shape
    reshaped = cube.reshape(n_frames, -1)

    # Fit PCA to model quasi-static stellar component
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(reshaped)
    
    # Transform and inverse transform to get PCA model
    transformed = pca.transform(reshaped)
    model = pca.inverse_transform(transformed)
    
    # Compute residuals (planet + noise)
    residuals = (reshaped - model).reshape(n_frames, ny, nx)
    pca_model = model.reshape(n_frames, ny, nx)

    # De-rotate each residual frame by negative parallactic angle
    derotated = np.zeros_like(residuals)
    for i in range(n_frames):
        derotated[i] = ndrotate(residuals[i], -angles[i], reshape=False, order=3)

    # Mean combine (planet coherent, noise averages)
    final_image = np.mean(derotated, axis=0)
    
    result = {
        "final_image": final_image,
        "derotated": derotated,
        "residuals": residuals,
        "pca_model": pca_model,
    }
    
    return result
