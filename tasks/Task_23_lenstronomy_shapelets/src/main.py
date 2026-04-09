import numpy as np
import time
from lenstronomy.Util import image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet


def load_and_preprocess_data(
    numPix: int,
    deltaPix: float,
    background_rms: float,
    exp_time: float,
    fwhm: float,
    kwargs_lens: list,
    kwargs_source_true: list,
    lens_model_list: list,
    source_model_list_true: list,
    random_seed: int = 42
) -> dict:
    """
    Load and preprocess data by simulating a lensed complex source with noise.
    
    Returns a dictionary containing all necessary data and class instances.
    """
    np.random.seed(random_seed)
    
    # Transformation matrix: pixel coordinates -> angular coordinates
    transform_pix2angle = np.array([[-deltaPix, 0], [0, deltaPix]])
    
    # Calculate the RA/Dec of the pixel (0,0) such that the image is centered at (0,0)
    cx = (numPix - 1) / 2.0
    cy = (numPix - 1) / 2.0
    ra_at_xy_0 = -(transform_pix2angle[0, 0] * cx + transform_pix2angle[0, 1] * cy)
    dec_at_xy_0 = -(transform_pix2angle[1, 0] * cx + transform_pix2angle[1, 1] * cy)
    
    # Initialize data class with zeros
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
    kwargs_psf = {
        'psf_type': 'GAUSSIAN',
        'fwhm': fwhm,
        'pixel_size': deltaPix,
        'truncation': 3
    }
    psf_class = PSF(**kwargs_psf)
    
    # Define Lens Model
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    
    # Define True Source Model
    source_model_class_true = LightModel(light_model_list=source_model_list_true)
    
    # No Lens Light
    lens_light_model_list = []
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    # Numerics settings
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    # Create ImageModel for simulation
    imageModel = ImageModel(
        data_class, psf_class, lens_model_class, source_model_class_true,
        lens_light_model_class, kwargs_numerics=kwargs_numerics
    )
    
    # Simulate clean image
    image_sim = imageModel.image(kwargs_lens, kwargs_source_true, kwargs_lens_light=None, kwargs_ps=None)
    
    # Add Poisson Noise
    image_sim_counts = image_sim * exp_time
    image_sim_counts[image_sim_counts < 0] = 0
    poisson_counts = np.random.poisson(image_sim_counts)
    poisson = poisson_counts / exp_time
    poisson_noise = poisson - image_sim
    
    # Add Gaussian Background Noise
    bkg_noise = np.random.randn(*image_sim.shape) * background_rms
    
    # Combine to get noisy image
    image_noisy = image_sim + bkg_noise + poisson_noise
    
    # Update Data with Simulated Image
    kwargs_data['image_data'] = image_noisy
    data_class.update_data(image_noisy)
    
    return {
        'data_class': data_class,
        'psf_class': psf_class,
        'lens_model_class': lens_model_class,
        'kwargs_numerics': kwargs_numerics,
        'kwargs_lens': kwargs_lens,
        'image_data': image_noisy,
        'image_clean': image_sim,
        'numPix': numPix,
        'deltaPix': deltaPix,
        'background_rms': background_rms,
        'exp_time': exp_time
    }


def forward_operator(
    shapelet_coeffs: np.ndarray,
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    kwargs_lens: list,
    kwargs_source_template: list,
    kwargs_numerics: dict
) -> np.ndarray:
    """
    Forward operator: Given shapelet coefficients, compute the predicted lensed image.
    
    This implements the forward model: source -> lensing -> convolution with PSF -> image.
    """
    # Create source model for shapelets
    source_model_list = ['SHAPELETS']
    source_model_class = LightModel(light_model_list=source_model_list)
    
    # No lens light
    lens_light_model_class = LightModel(light_model_list=[])
    
    # Create ImageModel
    imageModel = ImageModel(
        data_class, psf_class, lens_model_class, source_model_class,
        lens_light_model_class, kwargs_numerics=kwargs_numerics
    )
    
    # Build kwargs_source with amplitudes
    n_max = kwargs_source_template[0]['n_max']
    beta = kwargs_source_template[0]['beta']
    center_x = kwargs_source_template[0]['center_x']
    center_y = kwargs_source_template[0]['center_y']
    
    # The ShapeletSet uses 'amp' as a list of coefficients
    kwargs_source = [{
        'n_max': n_max,
        'beta': beta,
        'center_x': center_x,
        'center_y': center_y,
        'amp': shapelet_coeffs
    }]
    
    # Compute predicted image
    y_pred = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
    
    return y_pred


def run_inversion(
    data_class: ImageData,
    psf_class: PSF,
    lens_model_class: LensModel,
    kwargs_lens: list,
    kwargs_numerics: dict,
    n_max: int,
    beta: float,
    center_x: float = 0.0,
    center_y: float = 0.0
) -> dict:
    """
    Run the linear inversion to reconstruct the source using shapelets.
    
    Uses lenstronomy's ImageLinearFit to solve for shapelet coefficients.
    """
    # Define Reconstruction Model (Shapelets)
    source_model_list_reconstruct = ['SHAPELETS']
    source_model_class_reconstruct = LightModel(light_model_list=source_model_list_reconstruct)
    
    # Initialize ImageLinearFit
    imageLinearFit = ImageLinearFit(
        data_class=data_class,
        psf_class=psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class_reconstruct,
        kwargs_numerics=kwargs_numerics
    )
    
    # Constraints for Shapelets (center and scale)
    kwargs_source_reconstruct = [{
        'n_max': n_max,
        'beta': beta,
        'center_x': center_x,
        'center_y': center_y
    }]
    
    start_time = time.time()
    
    # image_linear_solve returns: model_image, error_map, cov_param, param
    wls_model, error_map, cov_param, param = imageLinearFit.image_linear_solve(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source_reconstruct,
        kwargs_lens_light=None,
        kwargs_ps=None,
        inv_bool=False
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate Reduced Chi-Square
    chi2_reduced = imageLinearFit.reduced_chi2(wls_model, error_map)
    
    # Extract Shapelet Coefficients
    shapelet_coeffs = np.array(param)
    
    return {
        'model_image': wls_model,
        'error_map': error_map,
        'cov_param': cov_param,
        'shapelet_coeffs': shapelet_coeffs,
        'chi2_reduced': chi2_reduced,
        'elapsed_time': elapsed_time,
        'kwargs_source_reconstruct': kwargs_source_reconstruct,
        'n_max': n_max,
        'beta': beta
    }


def evaluate_results(
    inversion_result: dict,
    data_dict: dict
) -> dict:
    """
    Evaluate the results of the inversion.
    
    Computes metrics and returns summary statistics.
    """
    model_image = inversion_result['model_image']
    shapelet_coeffs = inversion_result['shapelet_coeffs']
    chi2_reduced = inversion_result['chi2_reduced']
    elapsed_time = inversion_result['elapsed_time']
    n_max = inversion_result['n_max']
    beta = inversion_result['beta']
    
    image_data = data_dict['image_data']
    background_rms = data_dict['background_rms']
    
    # Compute residuals
    residuals = image_data - model_image
    
    # Compute RMS of residuals
    residual_rms = np.sqrt(np.mean(residuals**2))
    
    # Compute peak signal-to-noise ratio
    signal_max = np.max(np.abs(model_image))
    psnr = 20 * np.log10(signal_max / residual_rms) if residual_rms > 0 else np.inf
    
    # Number of shapelet coefficients
    num_coeffs = len(shapelet_coeffs)
    
    # Expected number of coefficients for given n_max
    expected_num_coeffs = int((n_max + 1) * (n_max + 2) / 2)
    
    # Compute coefficient statistics
    coeff_mean = np.mean(shapelet_coeffs)
    coeff_std = np.std(shapelet_coeffs)
    coeff_max = np.max(np.abs(shapelet_coeffs))
    
    # Print evaluation results
    print(f"Reconstruction completed in {elapsed_time:.4f} seconds.")
    print(f"Reduced Chi^2: {chi2_reduced:.4f}")
    print(f"Number of Shapelet coefficients: {num_coeffs}")
    print(f"Expected coefficients for n_max={n_max}: {expected_num_coeffs}")
    print(f"Residual RMS: {residual_rms:.6f}")
    print(f"Background RMS: {background_rms:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Shapelet beta (scale): {beta}")
    print(f"Coefficient mean: {coeff_mean:.6f}")
    print(f"Coefficient std: {coeff_std:.6f}")
    print(f"Coefficient max abs: {coeff_max:.6f}")
    
    return {
        'residuals': residuals,
        'residual_rms': residual_rms,
        'psnr': psnr,
        'chi2_reduced': chi2_reduced,
        'num_coeffs': num_coeffs,
        'elapsed_time': elapsed_time,
        'coeff_mean': coeff_mean,
        'coeff_std': coeff_std,
        'coeff_max': coeff_max
    }


if __name__ == '__main__':
    # Define parameters
    background_rms = 0.05
    exp_time = 100
    numPix = 100
    deltaPix = 0.05
    fwhm = 0.1
    
    # Lens model parameters
    lens_model_list = ['SIE', 'SHEAR']
    kwargs_sie = {'theta_E': 1.0, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0}
    kwargs_shear = {'gamma1': 0.05, 'gamma2': 0.01}
    kwargs_lens = [kwargs_sie, kwargs_shear]
    
    # True source model parameters
    source_model_list_true = ['SERSIC_ELLIPSE', 'SERSIC']
    kwargs_source_true = [
        {'amp': 200, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1},
        {'amp': 100, 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': -0.2, 'center_y': 0.0}
    ]
    
    # Shapelet reconstruction parameters
    n_max = 8
    beta = 0.2
    
    # Step 1: Load and preprocess data
    print("Data generation complete (Lensed Complex Source).")
    data_dict = load_and_preprocess_data(
        numPix=numPix,
        deltaPix=deltaPix,
        background_rms=background_rms,
        exp_time=exp_time,
        fwhm=fwhm,
        kwargs_lens=kwargs_lens,
        kwargs_source_true=kwargs_source_true,
        lens_model_list=lens_model_list,
        source_model_list_true=source_model_list_true,
        random_seed=42
    )
    
    # Step 2: Run inversion
    print("Starting linear inversion for source reconstruction...")
    inversion_result = run_inversion(
        data_class=data_dict['data_class'],
        psf_class=data_dict['psf_class'],
        lens_model_class=data_dict['lens_model_class'],
        kwargs_lens=data_dict['kwargs_lens'],
        kwargs_numerics=data_dict['kwargs_numerics'],
        n_max=n_max,
        beta=beta,
        center_x=0.0,
        center_y=0.0
    )
    
    # Step 3: Evaluate results
    evaluation = evaluate_results(inversion_result, data_dict)
    
    # Step 4: Demonstrate forward operator
    # Verify forward operator produces same result as inversion model
    y_pred = forward_operator(
        shapelet_coeffs=inversion_result['shapelet_coeffs'],
        data_class=data_dict['data_class'],
        psf_class=data_dict['psf_class'],
        lens_model_class=data_dict['lens_model_class'],
        kwargs_lens=data_dict['kwargs_lens'],
        kwargs_source_template=inversion_result['kwargs_source_reconstruct'],
        kwargs_numerics=data_dict['kwargs_numerics']
    )
    
    forward_diff = np.max(np.abs(y_pred - inversion_result['model_image']))
    print(f"Forward operator consistency check (max diff): {forward_diff:.2e}")
    
    print("Reconstruction successful.")
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")