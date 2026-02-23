# Lenstronomy Task: Quasar Host Galaxy Decomposition

This task demonstrates how to decompose a quasar (point source) and its host galaxy (extended source) from an image.
Lensing is ignored in this example; it focuses on galaxy photometry and deblending using lenstronomy's modeling capabilities.
The host galaxy is modeled as a composite of a disk (Exponential) and a bulge (De Vaucouleurs/Sersic n=4).

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
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
import lenstronomy.Util.param_util as param_util

# =============================================================================
# 1. Data Generation (Simulate Quasar + Host)
# =============================================================================
# Data specifics
background_rms = 0.1  # background noise
exp_time = 100.
numPix = 80
deltaPix = 0.05
fwhm = 0.1
psf_type = 'GAUSSIAN'

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
kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
psf_class = PSF(**kwargs_psf)

# Define Model (Unlensed)
# Quasar (Point Source)
center_x = 0.02
center_y = 0.01
point_amp = 10000
point_source_list = ['UNLENSED']
pointSource = PointSource(point_source_type_list=point_source_list)
kwargs_ps = [{'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]}]

# Host Galaxy (Disk + Bulge)
light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
lightModel = LightModel(light_model_list=light_model_list)
e1, e2 = param_util.phi_q2_ellipticity(phi=0.3, q=0.6)
kwargs_disk = {'amp': 400, 'n_sersic': 1, 'R_sersic': 0.7, 'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y}
kwargs_bulge = {'amp': 400, 'n_sersic': 4, 'R_sersic': 0.3, 'center_x': center_x, 'center_y': center_y}
kwargs_host = [kwargs_disk, kwargs_bulge]

# Simulate Image
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                        point_source_class=pointSource, kwargs_numerics=kwargs_numerics)

# --- Image Simulation and Noise Injection (Explicit) ---
image_sim = imageModel.image(kwargs_lens_light=kwargs_host, kwargs_ps=kwargs_ps)

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

# Update Data
kwargs_data['image_data'] = image_sim
data_class.update_data(image_sim)
print("Data generation complete (Quasar + Host).")

# =============================================================================
# 2. Model Fitting
# =============================================================================

# Define models for fitting
kwargs_model = { 'lens_light_model_list': light_model_list,
                'point_source_model_list': point_source_list
                 }

# Constraints
# Joint center for all components
kwargs_constraints = {'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y']]],
                      'joint_lens_light_with_point_source': [[0, 0], [0, 1]],
                      'num_point_source_list': [1]
                     }

kwargs_likelihood = {'check_bounds': True, 'source_marg': False}

image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

# Initial Params
# Host
fixed_source = []
kwargs_source_init = []
kwargs_source_sigma = []
kwargs_lower_source = []
kwargs_upper_source = []

# Disk (n=1 fixed)
fixed_source.append({'n_sersic': 1})
kwargs_source_init.append({'R_sersic': 1., 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10})
kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})

# Bulge (n=4 fixed)
fixed_source.append({'n_sersic': 4})
kwargs_source_init.append({'R_sersic': .5, 'n_sersic': 4, 'center_x': 0, 'center_y': 0})
kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.3, 'center_x': 0.1, 'center_y': 0.1})
kwargs_lower_source.append({'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10})
kwargs_upper_source.append({'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10})

source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]

# Point Source
fixed_ps = [{}]
kwargs_ps_init = [{'ra_image': [0.0], 'dec_image': [0.0]}]
kwargs_ps_sigma = [{'ra_image': [0.01], 'dec_image': [0.01]}]
kwargs_lower_ps = [{'ra_image': [-10], 'dec_image': [-10]}]
kwargs_upper_ps = [{'ra_image': [10], 'dec_image': [10]}]

ps_param = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

kwargs_params = {'lens_light_model': source_params,
               'point_source_model': ps_param}

# Fitting Sequence
fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50}]]

print("Starting fitting sequence...")
start_time = time.time()
chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()
end_time = time.time()
print(f"Fitting completed in {end_time - start_time:.2f} seconds.")

print("Best fit result:")
print(kwargs_result)

# Verify Fluxes
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
imageLinearFit = ImageLinearFit(data_class=data_class, psf_class=psf_class,
                        lens_light_model_class=lightModel,
                        point_source_class=pointSource, 
                        kwargs_numerics=kwargs_numerics)
image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(kwargs_lens_light=kwargs_result['kwargs_lens_light'], kwargs_ps=kwargs_result['kwargs_ps'])
print("Reconstructed Image Sum:", np.sum(image_reconstructed))
print("True Image Sum:", np.sum(image_sim))
```
