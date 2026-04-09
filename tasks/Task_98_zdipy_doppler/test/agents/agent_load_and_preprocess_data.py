import numpy as np

import matplotlib

matplotlib.use("Agg")

def load_and_preprocess_data(n_lat, n_lon, n_phases, n_vbins, v_max, v_eq,
                              inclination, local_width, limb_dark, snr_db):
    """
    Create surface grid, ground truth brightness map, observation setup,
    and build the design matrix.

    Returns
    -------
    data : dict containing all preprocessed data
    """
    # ── Surface grid ──
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_edges = np.linspace(0, 2 * np.pi, n_lon + 1)

    lats_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons_1d = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    LONS, LATS = np.meshgrid(lons_1d, lats_1d)
    lats = LATS.ravel()
    lons = LONS.ravel()

    d_lat = lat_edges[1] - lat_edges[0]
    d_lon = lon_edges[1] - lon_edges[0]
    d_omega = np.abs(np.cos(lats)) * d_lat * d_lon

    # ── Ground truth brightness map (3 dark spots) ──
    n_pix = n_lat * n_lon
    B_gt = np.ones(n_pix, dtype=np.float64)

    # Spot 1: large cool spot near equator
    lat_c1, lon_c1 = np.radians(10), np.radians(90)
    r1, depth1 = np.radians(20), 0.3
    ang1 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c1) +
        np.cos(lats) * np.cos(lat_c1) * np.cos(lons - lon_c1), -1, 1))
    B_gt[ang1 < r1] = depth1

    # Spot 2: mid-latitude spot
    lat_c2, lon_c2 = np.radians(40), np.radians(220)
    r2, depth2 = np.radians(15), 0.4
    ang2 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c2) +
        np.cos(lats) * np.cos(lat_c2) * np.cos(lons - lon_c2), -1, 1))
    B_gt[ang2 < r2] = depth2

    # Spot 3: small polar spot
    lat_c3, lon_c3 = np.radians(65), np.radians(350)
    r3, depth3 = np.radians(12), 0.5
    ang3 = np.arccos(np.clip(
        np.sin(lats) * np.sin(lat_c3) +
        np.cos(lats) * np.cos(lat_c3) * np.cos(lons - lon_c3), -1, 1))
    B_gt[ang3 < r3] = depth3

    # ── Observation setup ──
    phases = np.linspace(0, 1.0, n_phases, endpoint=False)
    v_axis = np.linspace(-v_max, v_max, n_vbins)

    # ── Build design matrix ──
    inc_rad = np.radians(inclination)
    A = np.zeros((n_phases * n_vbins, n_pix), dtype=np.float64)

    for ip, phi in enumerate(phases):
        # Compute visibility and velocity for this phase
        shifted_lon = lons + 2.0 * np.pi * phi
        sin_i = np.sin(inc_rad)
        cos_i = np.cos(inc_rad)
        mu = sin_i * np.cos(lats) * np.cos(shifted_lon) + cos_i * np.sin(lats)
        v_rad = v_eq * sin_i * np.cos(lats) * np.sin(shifted_lon)

        visible = mu > 0.0
        ld_weight = np.where(visible, 1.0 - limb_dark * (1.0 - mu), 0.0)
        weight = ld_weight * d_omega * visible.astype(np.float64)

        for j in range(n_pix):
            if not visible[j]:
                continue
            profile = np.exp(-0.5 * ((v_axis - v_rad[j]) / local_width) ** 2)
            profile /= (local_width * np.sqrt(2.0 * np.pi))
            A[ip * n_vbins:(ip + 1) * n_vbins, j] = weight[j] * profile

    # ── Forward solve to get clean data ──
    d_clean = A @ B_gt

    # ── Add noise ──
    noise_power = np.linalg.norm(d_clean) / (10 ** (snr_db / 20))
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(d_clean.shape)
    noise *= noise_power / np.linalg.norm(noise)
    d_noisy = d_clean + noise

    data = {
        'lats': lats,
        'lons': lons,
        'd_omega': d_omega,
        'B_gt': B_gt,
        'phases': phases,
        'v_axis': v_axis,
        'A': A,
        'd_clean': d_clean,
        'd_noisy': d_noisy,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'n_phases': n_phases,
        'n_vbins': n_vbins,
        'v_eq': v_eq,
        'inclination': inclination,
        'local_width': local_width,
        'limb_dark': limb_dark,
    }

    return data
