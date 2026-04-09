import numpy as np

import matplotlib

matplotlib.use("Agg")

def electron_wavelength_angstrom_local(E_eV):
    """Relativistic de Broglie wavelength [Å]."""
    import math as ma
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458.0
    h = 6.62607e-34
    lam = (h / ma.sqrt(2 * m * e * E_eV)
           / ma.sqrt(1 + e * E_eV / 2 / m / c**2) * 1e10)
    return lam

def run_inversion(datacube, energy, semiangle, scan_step, sampling, diff_px,
                  num_iter, step_size, max_batch, angular_sampling):
    """
    Run iterative ptychographic phase retrieval using py4DSTEM.
    
    This function performs gradient-descent based ptychographic reconstruction
    to recover the complex object from 4D-STEM diffraction data.
    
    Parameters
    ----------
    datacube : DataCube
        py4DSTEM DataCube containing 4D-STEM data
    energy : float
        Electron energy in eV
    semiangle : float
        Convergence semi-angle in mrad
    scan_step : float
        Scan step size in Angstroms
    sampling : float
        Real-space pixel size in Angstroms/pixel
    diff_px : int
        Diffraction pattern size in pixels
    num_iter : int
        Number of reconstruction iterations
    step_size : float
        Gradient descent step size
    max_batch : int
        Maximum batch size for reconstruction
    angular_sampling : float
        Angular sampling in mrad/pixel
    
    Returns
    -------
    dict containing:
        - recon_object: reconstructed complex object array
        - recon_phase: extracted phase from reconstruction
        - final_error: final reconstruction error
        - fov_mask: field-of-view mask
        - ptycho: the SingleslicePtychography instance
    """
    from py4DSTEM.process.phase import SingleslicePtychography
    from scipy.ndimage import gaussian_filter as gf

    print("\n[5/6] Running ptychographic reconstruction ...")

    wavelength = electron_wavelength_angstrom_local(energy)

    print(f"  Angular sampling = {angular_sampling:.4f} mrad/pixel")
    print(f"  Wavelength       = {wavelength:.5f} Å")

    ptycho = SingleslicePtychography(
        energy=energy,
        datacube=datacube,
        semiangle_cutoff=semiangle,
        device="cpu",
        verbose=True,
        object_type="potential",
    )

    ptycho.preprocess(
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        force_com_rotation=0.0,
        force_com_transpose=False,
        force_scan_sampling=scan_step,
        force_angular_sampling=angular_sampling,
    )

    ptycho.reconstruct(
        num_iter=num_iter,
        reconstruction_method="gradient-descent",
        step_size=step_size,
        max_batch_size=max_batch,
        fix_probe=False,
        fix_positions=True,
        progress_bar=True,
        reset=True,
        gaussian_filter_sigma=0.3,
        gaussian_filter=True,
        butterworth_filter=False,
        tv_denoise=False,
        object_positivity=True,
    )

    recon_object = np.array(ptycho.object)
    
    # Extract phase
    if np.iscomplexobj(recon_object):
        recon_phase = np.angle(recon_object)
    else:
        recon_phase = recon_object

    # Post-reconstruction Gaussian smoothing
    recon_phase = gf(recon_phase, sigma=1.0)

    print(f"  Recon object shape = {recon_object.shape}")
    print(f"  Recon phase range  = [{recon_phase.min():.4f}, "
          f"{recon_phase.max():.4f}]")
    print(f"  Final error        = {ptycho.error:.6f}")

    return {
        'recon_object': recon_object,
        'recon_phase': recon_phase,
        'final_error': ptycho.error,
        'fov_mask': ptycho._object_fov_mask,
        'ptycho': ptycho,
    }
