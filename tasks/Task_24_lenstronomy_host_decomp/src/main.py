import numpy as np
import time
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


def load_and_preprocess_data(numPix, deltaPix, background_rms, exp_time, fwhm, psf_type,
                              center_x, center_y, point_amp, kwargs_disk, kwargs_bulge):
    """
    Load and preprocess data: set up coordinate system, PSF, light models,
    simulate image with noise, and return all necessary components.
    
    Returns:
        dict containing:
            - data_class: ImageData object
            - psf_class: PSF object
            - lightModel: LightModel object
            - pointSource: PointSource object
            - image_sim: simulated image with noise
            - kwargs_data: data configuration
            - kwargs_psf: PSF configuration
            - kwargs_numerics: numerics configuration
            - kwargs_host: host galaxy parameters (ground truth)
            - kwargs_ps: point source parameters (ground truth)
            - light_model_list: list of light model names
            - point_source_list: list of point source model names
    """
    # Transformation matrix: pixel coordinates -> angular coordinates
    transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.
    cy = (numPix - 1) / 2.
    ra_at_xy_0 = - (transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = - (transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
    kwargs_data = {
        'background_rms': background_rms,
        'exposure_time': exp_time,
        'ra_at_xy_0': ra_at_xy_0,
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': transform_pix2angle,
        'image_data': np.zeros((numPix, numPix))
    }
    data_class = ImageData(**kwargs_data)
    
    # Configure PSF
    kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
    psf_class = PSF(**kwargs_psf)
    
    # Define Point Source Model
    point_source_list = ['UNLENSED']
    pointSource = PointSource(point_source_type_list=point_source_list)
    kwargs_ps = [{'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]}]
    
    # Define Host Galaxy Model (Disk + Bulge)
    light_model_list = ['SERSIC_ELLIPSE', 'SERSIC']
    lightModel = LightModel(light_model_list=light_model_list)
    kwargs_host = [kwargs_disk, kwargs_bulge]
    
    # Numerics configuration
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    # Create image model for simulation
    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                            point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
    
    # Simulate clean image
    image_sim = imageModel.image(kwargs_lens_light=kwargs_host, kwargs_ps=kwargs_ps)
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0
    poisson_counts = np.random.poisson(image_sim_counts)
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine noise
    image_sim = image_sim + bkg_noise + poisson_noise
    
    # Update Data
    kwargs_data['image_data'] = image_sim
    data_class.update_data(image_sim)
    
    return {
        'data_class': data_class,
        'psf_class': psf_class,
        'lightModel': lightModel,
        'pointSource': pointSource,
        'image_sim': image_sim,
        'kwargs_data': kwargs_data,
        'kwargs_psf': kwargs_psf,
        'kwargs_numerics': kwargs_numerics,
        'kwargs_host': kwargs_host,
        'kwargs_ps': kwargs_ps,
        'light_model_list': light_model_list,
        'point_source_list': point_source_list
    }


def forward_operator(data_class, psf_class, lightModel, pointSource, kwargs_numerics,
                     kwargs_lens_light, kwargs_ps):
    """
    Forward operator: given model parameters, compute the predicted image.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_numerics: numerics configuration
        kwargs_lens_light: lens light (host galaxy) parameters
        kwargs_ps: point source parameters
    
    Returns:
        y_pred: predicted image as numpy array
    """
    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                            point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
    y_pred = imageModel.image(kwargs_lens_light=kwargs_lens_light, kwargs_ps=kwargs_ps)
    return y_pred


def run_inversion(data_class, psf_class, lightModel, pointSource, kwargs_data, kwargs_psf,
                  kwargs_numerics, light_model_list, point_source_list,
                  n_particles=50, n_iterations=50):
    """
    Run the inversion/fitting process using PSO optimization.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_data: data configuration
        kwargs_psf: PSF configuration
        kwargs_numerics: numerics configuration
        light_model_list: list of light model names
        point_source_list: list of point source model names
        n_particles: number of PSO particles
        n_iterations: number of PSO iterations
    
    Returns:
        dict containing:
            - kwargs_result: best fit parameters
            - chain_list: fitting chain
            - fitting_time: time taken for fitting
    """
    # Define models for fitting
    kwargs_model = {
        'lens_light_model_list': light_model_list,
        'point_source_model_list': point_source_list
    }
    
    # Constraints: Joint center for all components
    kwargs_constraints = {
        'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y']]],
        'joint_lens_light_with_point_source': [[0, 0], [0, 1]],
        'num_point_source_list': [1]
    }
    
    kwargs_likelihood = {'check_bounds': True, 'source_marg': False}
    
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
    # Initial Parameters for Host Galaxy
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
    
    # Point Source Parameters
    fixed_ps = [{}]
    kwargs_ps_init = [{'ra_image': [0.0], 'dec_image': [0.0]}]
    kwargs_ps_sigma = [{'ra_image': [0.01], 'dec_image': [0.01]}]
    kwargs_lower_ps = [{'ra_image': [-10], 'dec_image': [-10]}]
    kwargs_upper_ps = [{'ra_image': [10], 'dec_image': [10]}]
    
    ps_param = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    
    kwargs_params = {
        'lens_light_model': source_params,
        'point_source_model': ps_param
    }
    
    # Fitting Sequence
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                   kwargs_likelihood, kwargs_params)
    
    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_particles, 'n_iterations': n_iterations}]]
    
    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    end_time = time.time()
    fitting_time = end_time - start_time
    
    return {
        'kwargs_result': kwargs_result,
        'chain_list': chain_list,
        'fitting_time': fitting_time
    }


def evaluate_results(data_class, psf_class, lightModel, pointSource, kwargs_numerics,
                     kwargs_result, image_sim):
    """
    Evaluate the fitting results by computing the reconstructed image and comparing
    with the observed data.
    
    Args:
        data_class: ImageData object
        psf_class: PSF object
        lightModel: LightModel object
        pointSource: PointSource object
        kwargs_numerics: numerics configuration
        kwargs_result: best fit parameters from inversion
        image_sim: observed/simulated image
    
    Returns:
        dict containing:
            - image_reconstructed: reconstructed image
            - residual: difference between observed and reconstructed
            - reconstructed_sum: sum of reconstructed image
            - true_sum: sum of observed image
            - residual_rms: RMS of residuals
    """
    imageLinearFit = ImageLinearFit(
        data_class=data_class,
        psf_class=psf_class,
        lens_light_model_class=lightModel,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics
    )
    
    image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(
        kwargs_lens_light=kwargs_result['kwargs_lens_light'],
        kwargs_ps=kwargs_result['kwargs_ps']
    )
    
    residual = image_sim - image_reconstructed
    reconstructed_sum = np.sum(image_reconstructed)
    true_sum = np.sum(image_sim)
    residual_rms = np.sqrt(np.mean(residual**2))
    
    return {
        'image_reconstructed': image_reconstructed,
        'residual': residual,
        'reconstructed_sum': reconstructed_sum,
        'true_sum': true_sum,
        'residual_rms': residual_rms
    }


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Data specifics
    background_rms = 0.1
    exp_time = 100.
    numPix = 80
    deltaPix = 0.05
    fwhm = 0.1
    psf_type = 'GAUSSIAN'
    
    # Model parameters
    center_x = 0.02
    center_y = 0.01
    point_amp = 10000
    
    # Host galaxy parameters
    e1, e2 = param_util.phi_q2_ellipticity(phi=0.3, q=0.6)
    kwargs_disk = {
        'amp': 400, 'n_sersic': 1, 'R_sersic': 0.7,
        'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y
    }
    kwargs_bulge = {
        'amp': 400, 'n_sersic': 4, 'R_sersic': 0.3,
        'center_x': center_x, 'center_y': center_y
    }
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(
        numPix=numPix,
        deltaPix=deltaPix,
        background_rms=background_rms,
        exp_time=exp_time,
        fwhm=fwhm,
        psf_type=psf_type,
        center_x=center_x,
        center_y=center_y,
        point_amp=point_amp,
        kwargs_disk=kwargs_disk,
        kwargs_bulge=kwargs_bulge
    )
    print("Data generation complete (Quasar + Host).")
    
    # Step 2: Demonstrate forward operator
    print("\nTesting forward operator...")
    y_pred = forward_operator(
        data_class=data['data_class'],
        psf_class=data['psf_class'],
        lightModel=data['lightModel'],
        pointSource=data['pointSource'],
        kwargs_numerics=data['kwargs_numerics'],
        kwargs_lens_light=data['kwargs_host'],
        kwargs_ps=data['kwargs_ps']
    )
    print(f"Forward operator output shape: {y_pred.shape}")
    print(f"Forward operator output sum: {np.sum(y_pred):.2f}")
    
    # Step 3: Run inversion
    print("\nStarting fitting sequence...")
    inversion_result = run_inversion(
        data_class=data['data_class'],
        psf_class=data['psf_class'],
        lightModel=data['lightModel'],
        pointSource=data['pointSource'],
        kwargs_data=data['kwargs_data'],
        kwargs_psf=data['kwargs_psf'],
        kwargs_numerics=data['kwargs_numerics'],
        light_model_list=data['light_model_list'],
        point_source_list=data['point_source_list'],
        n_particles=50,
        n_iterations=50
    )
    print(f"Fitting completed in {inversion_result['fitting_time']:.2f} seconds.")
    print("\nBest fit result:")
    print(inversion_result['kwargs_result'])
    
    # Step 4: Evaluate results
    print("\nEvaluating results...")
    evaluation = evaluate_results(
        data_class=data['data_class'],
        psf_class=data['psf_class'],
        lightModel=data['lightModel'],
        pointSource=data['pointSource'],
        kwargs_numerics=data['kwargs_numerics'],
        kwargs_result=inversion_result['kwargs_result'],
        image_sim=data['image_sim']
    )
    print(f"Reconstructed Image Sum: {evaluation['reconstructed_sum']:.2f}")
    print(f"True Image Sum: {evaluation['true_sum']:.2f}")
    print(f"Residual RMS: {evaluation['residual_rms']:.4f}")
    
    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")