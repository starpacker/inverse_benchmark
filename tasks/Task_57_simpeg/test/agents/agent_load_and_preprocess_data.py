import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_CELLS_X = 20

N_CELLS_Y = 20

N_CELLS_Z = 10

CELL_SIZE_X = 50.0

CELL_SIZE_Y = 50.0

CELL_SIZE_Z = 25.0

N_RX_X = 15

N_RX_Y = 15

RX_HEIGHT = 1.0

NOISE_FLOOR = 0.01

NOISE_PCT = 0.02

GT_DENSITY = 0.5

GT_CENTER = [0.0, 0.0, -150.0]

GT_RADIUS = 100.0

SEED = 42

GRAV_CONST = 6.674e-3

def create_mesh_info():
    """Create mesh information dictionary."""
    hx = np.ones(N_CELLS_X) * CELL_SIZE_X
    hy = np.ones(N_CELLS_Y) * CELL_SIZE_Y
    hz = np.ones(N_CELLS_Z) * CELL_SIZE_Z
    
    # Origin so that surface is at z=0, mesh extends downward
    origin = np.array([
        -N_CELLS_X * CELL_SIZE_X / 2,
        -N_CELLS_Y * CELL_SIZE_Y / 2,
        -N_CELLS_Z * CELL_SIZE_Z,
    ])
    
    # Compute cell centers
    x_centers = origin[0] + np.cumsum(hx) - hx / 2
    y_centers = origin[1] + np.cumsum(hy) - hy / 2
    z_centers = origin[2] + np.cumsum(hz) - hz / 2
    
    # Create 3D grid of cell centers (Fortran order for consistency)
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    cell_centers = np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
    
    mesh_info = {
        'hx': hx,
        'hy': hy,
        'hz': hz,
        'origin': origin,
        'shape_cells': (N_CELLS_X, N_CELLS_Y, N_CELLS_Z),
        'n_cells': N_CELLS_X * N_CELLS_Y * N_CELLS_Z,
        'cell_centers': cell_centers,
        'cell_volumes': CELL_SIZE_X * CELL_SIZE_Y * CELL_SIZE_Z,
    }
    return mesh_info

def create_receiver_locations():
    """Create surface gravity receiver locations."""
    rx_x = np.linspace(
        -N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_CELLS_X * CELL_SIZE_X / 2 * 0.7,
        N_RX_X
    )
    rx_y = np.linspace(
        -N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_CELLS_Y * CELL_SIZE_Y / 2 * 0.7,
        N_RX_Y
    )
    rx_xx, rx_yy = np.meshgrid(rx_x, rx_y)
    rx_locs = np.c_[
        rx_xx.ravel(),
        rx_yy.ravel(),
        np.full(N_RX_X * N_RX_Y, RX_HEIGHT)
    ]
    return rx_locs

def create_ground_truth(mesh_info):
    """Create ground truth density model: spherical anomaly."""
    cc = mesh_info['cell_centers']
    dist = np.sqrt(
        (cc[:, 0] - GT_CENTER[0]) ** 2 +
        (cc[:, 1] - GT_CENTER[1]) ** 2 +
        (cc[:, 2] - GT_CENTER[2]) ** 2
    )
    model_gt = np.zeros(mesh_info['n_cells'])
    model_gt[dist < GT_RADIUS] = GT_DENSITY

    # Add a smaller secondary anomaly
    dist2 = np.sqrt(
        (cc[:, 0] - (GT_CENTER[0] + 200)) ** 2 +
        (cc[:, 1] - (GT_CENTER[1] - 150)) ** 2 +
        (cc[:, 2] - (GT_CENTER[2] - 50)) ** 2
    )
    model_gt[dist2 < GT_RADIUS * 0.6] = -0.3

    return model_gt

def prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho):
    """
    Compute vertical gravity component gz for a rectangular prism.
    
    Uses the analytical formula from Blakely (1996) / Nagy (1966).
    
    Parameters
    ----------
    x1, x2 : float  Prism x bounds
    y1, y2 : float  Prism y bounds
    z1, z2 : float  Prism z bounds (z positive downward in convention, but we use z negative for depth)
    xp, yp, zp : float  Observation point coordinates
    rho : float  Density contrast [g/cm³]
    
    Returns
    -------
    gz : float  Vertical gravity component [mGal]
    """
    # Shift coordinates relative to observation point
    dx = [x1 - xp, x2 - xp]
    dy = [y1 - yp, y2 - yp]
    dz = [z1 - zp, z2 - zp]
    
    gz = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                r = np.sqrt(x**2 + y**2 + z**2)
                
                # Avoid singularities
                r = max(r, 1e-10)
                
                # Sign for the sum
                sign = (-1) ** (i + j + k)
                
                # Compute terms
                term1 = 0.0
                term2 = 0.0
                term3 = 0.0
                
                # x * ln(y + r)
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                # y * ln(x + r)
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                # z * arctan(xy / (zr))
                denom = z * r
                if abs(denom) > 1e-10:
                    term3 = z * np.arctan2(x * y, denom)
                
                gz += sign * (term1 + term2 - term3)
    
    # Convert to mGal (G in appropriate units)
    # G = 6.674e-11 m³/(kg·s²), density in g/cm³ = 1000 kg/m³
    # 1 mGal = 1e-5 m/s²
    # gz = G * rho * integral, with rho in g/cm³
    # Factor: 6.674e-11 * 1000 * 1e5 = 6.674e-3
    gz *= GRAV_CONST * rho
    
    return gz

def load_and_preprocess_data():
    """
    Generate synthetic gravity survey data.
    
    Returns
    -------
    data_dict : dict containing:
        - mesh_info: mesh information
        - rx_locs: receiver locations
        - model_gt: ground truth density model
        - d_clean: clean gravity data
        - d_noisy: noisy gravity data
        - std: data standard deviation
    """
    print("[DATA] Creating 3D tensor mesh ...")
    mesh_info = create_mesh_info()
    print(f"[DATA] Mesh: {mesh_info['shape_cells']} cells")

    print("[DATA] Creating gravity survey ...")
    rx_locs = create_receiver_locations()
    print(f"[DATA] {rx_locs.shape[0]} receivers at z={RX_HEIGHT} m")

    print("[DATA] Building ground truth density model ...")
    model_gt = create_ground_truth(mesh_info)
    print(f"[DATA] Anomaly: {(model_gt != 0).sum()} active cells, "
          f"Δρ range [{model_gt.min():.2f}, {model_gt.max():.2f}] g/cm³")

    print("[DATA] Running forward simulation ...")
    d_clean = forward_operator(model_gt, mesh_info, rx_locs)
    print(f"[DATA] g_z range: [{d_clean.min():.4f}, {d_clean.max():.4f}] mGal")

    # Add noise
    rng = np.random.default_rng(SEED)
    std = NOISE_FLOOR + NOISE_PCT * np.abs(d_clean)
    noise = std * rng.standard_normal(len(d_clean))
    d_noisy = d_clean + noise

    data_dict = {
        'mesh_info': mesh_info,
        'rx_locs': rx_locs,
        'model_gt': model_gt,
        'd_clean': d_clean,
        'd_noisy': d_noisy,
        'std': std,
    }
    
    return data_dict

def forward_operator(model, mesh_info, rx_locs):
    """
    Gravity forward simulation using prism formula.
    
    Computes vertical gravity component gz at each receiver location
    by summing contributions from all mesh cells (treated as uniform
    density prisms).
    
    Parameters
    ----------
    model : np.ndarray  Density contrast vector (g/cm³), shape (n_cells,)
    mesh_info : dict    Mesh information
    rx_locs : np.ndarray  Receiver locations, shape (n_rx, 3)
    
    Returns
    -------
    d_pred : np.ndarray  Predicted gravity anomaly [mGal], shape (n_rx,)
    """
    n_rx = rx_locs.shape[0]
    n_cells = mesh_info['n_cells']
    cc = mesh_info['cell_centers']
    
    # Half cell sizes
    dx = mesh_info['hx'][0] / 2
    dy = mesh_info['hy'][0] / 2
    dz = mesh_info['hz'][0] / 2
    
    d_pred = np.zeros(n_rx)
    
    # Only process cells with non-zero density for efficiency
    active_cells = np.where(model != 0)[0]
    
    for i_rx in range(n_rx):
        xp, yp, zp = rx_locs[i_rx]
        gz_total = 0.0
        
        for i_cell in active_cells:
            xc, yc, zc = cc[i_cell]
            rho = model[i_cell]
            
            # Prism bounds
            x1, x2 = xc - dx, xc + dx
            y1, y2 = yc - dy, yc + dy
            z1, z2 = zc - dz, zc + dz
            
            gz_total += prism_gz(x1, x2, y1, y2, z1, z2, xp, yp, zp, rho)
        
        d_pred[i_rx] = gz_total
    
    return d_pred
