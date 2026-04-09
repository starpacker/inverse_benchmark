import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_uv_coverage(n_stations, obs_hours, n_time):
    """Generate uv-coverage for an EHT-like VLBI array."""
    stations_lat = np.array([19.8, 37.1, -23.0, 32.7, -67.8, 78.2, 28.3, -30.7])[:n_stations]
    stations_lon = np.array([-155.5, -3.4, -67.8, -109.9, -68.8, 15.5, -16.6, 21.4])[:n_stations]
    lat_rad = np.deg2rad(stations_lat)
    lon_rad = np.deg2rad(stations_lon)
    wavelength_m = 1.3e-3
    earth_radius_m = 6.371e6
    R_lambda = earth_radius_m / wavelength_m
    X_st = R_lambda * np.cos(lat_rad) * np.cos(lon_rad)
    Y_st = R_lambda * np.cos(lat_rad) * np.sin(lon_rad)
    Z_st = R_lambda * np.sin(lat_rad)
    dec = np.deg2rad(12.0)
    ha = np.linspace(-obs_hours / 2, obs_hours / 2, n_time) * (np.pi / 12.0)
    u_all, v_all = [], []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            dx = X_st[j] - X_st[i]
            dy = Y_st[j] - Y_st[i]
            dz = Z_st[j] - Z_st[i]
            for h in ha:
                u = np.sin(h) * dx + np.cos(h) * dy
                v = (-np.sin(dec) * np.cos(h) * dx + np.sin(dec) * np.sin(h) * dy + np.cos(dec) * dz)
                u_all.append(u)
                v_all.append(v)
                u_all.append(-u)
                v_all.append(-v)
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    u_all *= uas_to_rad
    v_all *= uas_to_rad
    return u_all, v_all
