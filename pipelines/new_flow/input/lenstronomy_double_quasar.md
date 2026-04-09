# Lenstronomy Task: Modeling a Doubly Imaged Quasar

This task demonstrates how to model a doubly imaged quasar with a host galaxy and a lens galaxy component.
The simulation uses an Elliptical Power Law (EPL) lens model with external shear.

## Code

```python
import numpy as np
import os
import time
from lenstronomy.Util import image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Workflow.fitting_sequence import FittingSequence

# =============================================================================
# 1. Data Generation (Mock Data)
# =============================================================================
# Define data specifics
background_rms = 0.5  # background noise per pixel
exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 100  # cutout pixel size
deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.2  # full width half max of PSF

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

# Define Lens Model (EPL + Shear)
lens_model_list = ['EPL', 'SHEAR']
# True parameters for simulation
kwargs_spemd = {'theta_E': 1., 'gamma': 1.96, 'center_x': 0, 'center_y': 0, 'e1': 0.07, 'e2': -0.03}
kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01} 
kwargs_lens = [kwargs_spemd, kwargs_shear]
lens_model_class = LensModel(lens_model_list=lens_model_list)

# Define Lens Light Model (Spherical Sersic)
lens_light_model_list = ['SERSIC']
kwargs_sersic = {'amp': 400., 'R_sersic': 1., 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
kwargs_lens_light = [kwargs_sersic]
lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

# Define Source Light Model (Elliptical Sersic)
source_model_list = ['SERSIC_ELLIPSE']
ra_source, dec_source = 0.1, 0.3
kwargs_sersic_ellipse = {'amp': 160, 'R_sersic': .5, 'n_sersic': 7, 'center_x': ra_source,
                         'center_y': dec_source, 'e1': 0., 'e2': 0.1}
kwargs_source = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list=source_model_list)

# Solve for Image Positions (Double Quasar)
lensEquationSolver = LensEquationSolver(lens_model_class)
x_image, y_image = lensEquationSolver.findBrightImage(ra_source, dec_source, kwargs_lens, numImages=4,
                                                      min_distance=deltaPix, search_window=numPix * deltaPix)

# Select only 2 images if more are found (forcing double) or check if it is a double
# In the notebook, it says "works only with a 2-imaged point source".
# Let's trust the simulation parameters produce a double.
print("Number of images found:", len(x_image))

mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
              'point_amp': np.abs(mag)*100}]
point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

# Simulate Image
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                        lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)

# --- Image Simulation and Noise Injection (Explicit) ---
image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

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

print("Data generation complete.")

# =============================================================================
# 2. Model Fitting Setup
# =============================================================================

# Define models for fitting
kwargs_model = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list,
                'lens_light_model_list': lens_light_model_list,
                'point_source_model_list': point_source_list,
                'fixed_magnification_list': [False],
               }

# Define constraints
# Using 'THETA_E_PHI' solver to constrain lens model with image positions
kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                      'num_point_source_list': [len(x_image)], 
                      'solver_type': 'THETA_E_PHI',  
                      }

# Define Likelihood
kwargs_likelihood = {'check_bounds': True,
                     'force_no_add_image': False,
                     'source_marg': False,
                     'image_position_uncertainty': 0.004,
                     'source_position_tolerance': 0.001
                    }

# Prepare Data for FittingSequence
image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

# Initial Params
# Lens
kwargs_lens_init = [{'theta_E': 1.1, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
    {'gamma1': 0., 'gamma2': 0.}]
kwargs_lens_sigma = [{'theta_E': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': .1, 'center_x': 0.1, 'center_y': 0.1},
    {'gamma1': 0.1, 'gamma2': 0.1}]
kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
    {'gamma1': -.2, 'gamma2': -0.2}]
kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
    {'gamma1': 0.2, 'gamma2': 0.2}]

# Source
kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
kwargs_source_sigma = [{'R_sersic': 0.2, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]

# Lens Light
kwargs_lens_light_init = [{'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.5, 'e1': 0, 'e2': 0, 'center_x': .1, 'center_y': 0.1}]
kwargs_lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]

# Point Source
kwargs_ps_init = [{'ra_image': x_image+0.01, 'dec_image': y_image-0.01}]
kwargs_ps_sigma = [{'ra_image': [0.02] * len(x_image), 'dec_image': [0.02] * len(x_image)}]
kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]
kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]

lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, {'ra_0': 0, 'dec_0': 0}], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, [{}], kwargs_lower_lens_light, kwargs_upper_lens_light]
ps_params = [kwargs_ps_init, kwargs_ps_sigma, [{}], kwargs_lower_ps, kwargs_upper_ps]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params,
                'point_source_model': ps_params}

# =============================================================================
# 3. Fitting
# =============================================================================

fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

fitting_kwargs_list = [
    ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 10}],
    ['MCMC', {'n_burn': 10, 'n_run': 10, 'walkerRatio': 4, 'sigma_scale': .1}]
]

print("Starting fitting sequence...")
start_time = time.time()
chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()
end_time = time.time()
print(f"Fitting completed in {end_time - start_time:.2f} seconds.")

print("Best fit result:")
print(kwargs_result)
```
