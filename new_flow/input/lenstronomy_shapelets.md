# Lenstronomy Task: Shapelet Source Reconstruction

This task demonstrates how to reconstruct the source light distribution of a lensed object using Shapelet basis functions.
Unlike parametric models (e.g., Sersic), Shapelets can capture complex, irregular source morphologies.
The simulation creates a complex source (two Sersic clumps), lenses it, and then reconstructs it using linear inversion with Shapelets.

## Code

```python
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from lenstronomy.Util import image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

# =============================================================================
# 1. Data Generation (Simulate a Lensed Complex Source)
# =============================================================================
# Define data specifics
background_rms = 0.05  # background noise per pixel
exp_time = 100  # exposure time
numPix = 100  # cutout pixel size
deltaPix = 0.05  # pixel size in arcsec
fwhm = 0.1  # full width half max of PSF

# --- Explicit Data Generation ---
# Transformation matrix: pixel coordinates -> angular coordinates
# [-deltaPix, 0] means x-axis corresponds to RA decreasing (standard convention)
# [0, deltaPix] means y-axis corresponds to Dec increasing
transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])

# Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
# Center index in pixels
cx = (numPix - 1) / 2.
cy = (numPix - 1) / 2.
# Invert transformation to find origin
ra_at_xy_0 = - (transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
dec_at_xy_0 = - (transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)

kwargs_data = {
    'background_rms': background_rms,
    'exposure_time': exp_time,
    'ra_at_xy_0': ra_at_xy_0,
    'dec_at_xy_0': dec_at_xy_0,
    'transform_pix2angle': transform_pix2angle,
    'image_data': np.zeros((numPix, numPix)) # Initialize with zeros
}
data_class = ImageData(**kwargs_data)

# Configure PSF
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
psf_class = PSF(**kwargs_psf)

# Define Lens Model (SIE + Shear)
lens_model_list = ['SIE', 'SHEAR']
kwargs_sie = {'theta_E': 1.0, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0}
kwargs_shear = {'gamma1': 0.05, 'gamma2': 0.01}
kwargs_lens = [kwargs_sie, kwargs_shear]
lens_model_class = LensModel(lens_model_list=lens_model_list)

# Define Complex Source (Two Sersic Clumps to mimic structure)
source_model_list_true = ['SERSIC_ELLIPSE', 'SERSIC']
kwargs_source_true = [
    {'amp': 200, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
    {'amp': 100, 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': -0.2, 'center_y': 0.0}
]
source_model_class_true = LightModel(light_model_list=source_model_list_true)

# No Lens Light
lens_light_model_list = []
lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

# Simulate Image
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class_true,
                        lens_light_model_class, kwargs_numerics=kwargs_numerics)

# --- Image Simulation and Noise Injection (Explicit) ---
image_sim = imageModel.image(kwargs_lens, kwargs_source_true, kwargs_lens_light=None, kwargs_ps=None)

# Add Poisson Noise
image_sim_counts = image_sim * exp_time
image_sim_counts[image_sim_counts < 0] = 0
poisson_counts = np.random.poisson(image_sim_counts)
poisson = poisson_counts / exp_time
# Explicitly calculate noise map for demonstration (flux - mean)
poisson_noise = poisson - image_sim 

# Add Gaussian Background Noise
bkg_noise = np.random.randn(*image_sim.shape) * background_rms

# Combine
image_sim = image_sim + bkg_noise + poisson_noise

# Update Data with Simulated Image
kwargs_data['image_data'] = image_sim
data_class.update_data(image_sim)

print("Data generation complete (Lensed Complex Source).")

# =============================================================================
# 2. Source Reconstruction using Shapelets
# =============================================================================

# Define Reconstruction Model (Shapelets)
n_max = 8  # Order of Shapelets
beta = 0.2  # Scale of Shapelets
source_model_list_reconstruct = ['SHAPELETS']
source_model_class_reconstruct = LightModel(light_model_list=source_model_list_reconstruct)

# Initialize ImageLinearFit
# We fix the lens model to the truth and reconstruct the source
imageLinearFit = ImageLinearFit(data_class=data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                source_model_class=source_model_class_reconstruct, kwargs_numerics=kwargs_numerics)

# Constraints for Shapelets (center and scale)
# The amplitudes will be solved linearly
kwargs_source_reconstruct = [{'n_max': n_max, 'beta': beta, 'center_x': 0.0, 'center_y': 0.0}]

print("Starting linear inversion for source reconstruction...")
start_time = time.time()

# image_linear_solve returns: model_image, error_map, cov_param, param
# param contains the linear coefficients (amplitudes)
wls_model, error_map, cov_param, param = imageLinearFit.image_linear_solve(
    kwargs_lens=kwargs_lens,
    kwargs_source=kwargs_source_reconstruct,
    kwargs_lens_light=None,
    kwargs_ps=None,
    inv_bool=False
)

end_time = time.time()
print(f"Reconstruction completed in {end_time - start_time:.4f} seconds.")

# Calculate Reduced Chi-Square
chi2_reduced = imageLinearFit.reduced_chi2(wls_model, error_map)
print(f"Reduced Chi^2: {chi2_reduced:.4f}")

# Extract Shapelet Coefficients
shapelet_coeffs = param
print(f"Number of Shapelet coefficients: {len(shapelet_coeffs)}")

# Reconstruct Source in Source Plane
x_grid, y_grid = data_class.pixel_coordinates
shapeletSet = ShapeletSet()
# Note: shapeletSet.function takes 1D arrays
# We reconstruct on the same grid to visualize, but in principle source plane is distinct
# Here we just want to show we got coefficients.

print("Reconstruction successful.")
```
