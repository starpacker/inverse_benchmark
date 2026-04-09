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

def make_ground_truth_phase(shape):
    """
    Create a 2-D phase map with structured features.
    """
    H, W = shape
    y, x = np.mgrid[:H, :W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0

    phase = np.zeros((H, W), dtype=np.float64)
    rng = np.random.RandomState(42)

    n_peaks = 12
    for _ in range(n_peaks):
        px = rng.uniform(W * 0.15, W * 0.85)
        py = rng.uniform(H * 0.15, H * 0.85)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.15, 0.35)
        phase += amp * np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma**2))

    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    phase += 0.25 * np.exp(-((r - min(H, W) * 0.25) / 5.0)**2)

    phase = phase / (phase.max() + 1e-12) * 0.5
    return phase

def generate_synthetic_4dstem(gt_phase, probe_array, positions_px, dose):
    """
    Forward-simulate 4D-STEM data using the exact same coordinate
    convention that py4DSTEM's SingleslicePtychography uses internally.
    """
    from py4DSTEM.process.phase.utils import fft_shift

    Sx, Sy = probe_array.shape
    Px, Py = gt_phase.shape
    N = positions_px.shape[0]

    gt_object = np.exp(1j * gt_phase).astype(np.complex64)

    x_ind = np.fft.fftfreq(Sx, d=1.0 / Sx).astype(int)
    y_ind = np.fft.fftfreq(Sy, d=1.0 / Sy).astype(int)

    data = np.zeros((N, Sx, Sy), dtype=np.float64)

    for i in range(N):
        pos = positions_px[i]
        r0 = int(np.round(pos[0]))
        c0 = int(np.round(pos[1]))
        pos_frac = pos - np.round(pos)

        row_idx = (r0 + x_ind) % Px
        col_idx = (c0 + y_ind) % Py
        obj_patch = gt_object[np.ix_(row_idx, col_idx)]

        shifted_probe = fft_shift(
            probe_array, pos_frac.reshape(1, 2), np
        )[0]

        overlap = shifted_probe * obj_patch
        fourier_overlap = np.fft.fft2(overlap)
        dp_corner = np.abs(fourier_overlap) ** 2

        dp_centered = np.fft.fftshift(dp_corner)

        dp_scaled = dp_centered / (dp_centered.sum() + 1e-30) * dose
        data[i] = np.random.poisson(
            np.clip(dp_scaled, 0, None)
        ).astype(np.float64)

    return data.astype(np.float32)

def load_and_preprocess_data(energy, semiangle, scan_px, diff_px, scan_step, sampling, dose):
    """
    Load/generate and preprocess 4D-STEM data for ptychographic reconstruction.
    
    This function:
    1. Performs a dry-run to determine reconstruction geometry
    2. Creates ground-truth phase object
    3. Builds the electron probe
    4. Forward-simulates 4D-STEM diffraction data
    5. Packages data into a DataCube
    
    Parameters
    ----------
    energy : float
        Electron energy in eV
    semiangle : float
        Convergence semi-angle in mrad
    scan_px : int
        Number of scan positions per dimension
    diff_px : int
        Diffraction pattern size in pixels
    scan_step : float
        Scan step size in Angstroms
    sampling : float
        Real-space pixel size in Angstroms/pixel
    dose : float
        Total electron counts per diffraction pattern
    
    Returns
    -------
    dict containing:
        - datacube: py4DSTEM DataCube object
        - gt_phase: ground truth phase array
        - probe_array: complex probe array
        - positions_px: probe positions in pixels
        - obj_shape: reconstruction object shape
        - recon_sampling: reconstruction sampling
        - data_4d: raw 4D data array
        - angular_sampling: angular sampling in mrad/pixel
    """
    from py4DSTEM.process.phase.utils import ComplexProbe
    from py4DSTEM.datacube import DataCube as DC
    from py4DSTEM.process.phase import SingleslicePtychography

    print("\n[1/6] Determining reconstruction geometry (dry-run) ...")

    wavelength = electron_wavelength_angstrom_local(energy)
    angular_sampling = wavelength * 1e3 / (diff_px * sampling)

    # Dry-run to learn geometry
    dummy_data = np.ones(
        (scan_px, scan_px, diff_px, diff_px), dtype=np.float32
    )
    dummy_dc = DC(data=dummy_data)
    dummy_dc.calibration.set_R_pixel_size(scan_step)
    dummy_dc.calibration.set_R_pixel_units("A")
    dummy_dc.calibration.set_Q_pixel_size(1.0 / (diff_px * sampling))
    dummy_dc.calibration.set_Q_pixel_units("A^-1")

    dry = SingleslicePtychography(
        energy=energy,
        datacube=dummy_dc,
        semiangle_cutoff=semiangle,
        device="cpu",
        verbose=False,
        object_type="potential",
    )
    dry.preprocess(
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        force_com_rotation=0.0,
        force_com_transpose=False,
        force_scan_sampling=scan_step,
        force_angular_sampling=angular_sampling,
    )

    obj_shape = dry._object.shape
    positions_px = np.array(dry._positions_px)
    recon_sampling = dry.sampling

    print(f"  Reconstructed object will be {obj_shape}")
    print(f"  Real-space sampling = {recon_sampling[0]:.5f} Å/px")
    print(f"  # positions = {positions_px.shape[0]}")
    print(f"  Position range (px): "
          f"[{positions_px.min(0)[0]:.1f}–{positions_px.max(0)[0]:.1f}] × "
          f"[{positions_px.min(0)[1]:.1f}–{positions_px.max(0)[1]:.1f}]")

    del dry, dummy_dc, dummy_data

    # Create ground-truth phase
    print("\n[2/6] Creating ground-truth phase object ...")
    gt_phase = make_ground_truth_phase(obj_shape)
    print(f"  GT phase shape = {gt_phase.shape},  "
          f"range = [{gt_phase.min():.3f}, {gt_phase.max():.3f}] rad")

    # Build probe
    print("\n[3/6] Building electron probe ...")
    probe = ComplexProbe(
        energy=energy,
        gpts=(diff_px, diff_px),
        sampling=(recon_sampling[0], recon_sampling[1]),
        semiangle_cutoff=semiangle,
        device="cpu",
    )
    probe.build()
    probe_array = np.array(probe._array, dtype=np.complex128)
    print(f"  Probe shape = {probe_array.shape}, "
          f"|probe|² sum = {(np.abs(probe_array)**2).sum():.6f}")

    # Forward-simulate 4D-STEM data
    print("\n[4/6] Forward-simulating 4D-STEM diffraction data ...")
    np.random.seed(2024)
    flat_data = generate_synthetic_4dstem(
        gt_phase, probe_array, positions_px, dose
    )
    print(f"  Flat data shape  = {flat_data.shape}")
    print(f"  Mean counts/pat  = {flat_data.mean(axis=(1, 2)).mean():.1f}")

    # Reshape to 4D
    data_4d = flat_data.reshape(scan_px, scan_px, diff_px, diff_px)

    # Package as DataCube
    datacube = DC(data=data_4d)
    datacube.calibration.set_R_pixel_size(scan_step)
    datacube.calibration.set_R_pixel_units("A")
    datacube.calibration.set_Q_pixel_size(1.0 / (diff_px * sampling))
    datacube.calibration.set_Q_pixel_units("A^-1")

    print(f"  DataCube shape   = {data_4d.shape}")

    return {
        'datacube': datacube,
        'gt_phase': gt_phase,
        'probe_array': probe_array,
        'positions_px': positions_px,
        'obj_shape': obj_shape,
        'recon_sampling': recon_sampling,
        'data_4d': data_4d,
        'angular_sampling': angular_sampling,
    }
