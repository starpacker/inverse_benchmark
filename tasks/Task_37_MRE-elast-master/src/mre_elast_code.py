import os
import sys
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.io as sio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import triangle  # pip install triangle
from pyunlocbox import functions, solvers  # pip install pyunlocbox

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================

def load_and_preprocess_data(filename, target_size=64):
    """
    Loads an image, converts it to a mesh-based stiffness distribution (ground truth),
    and computes the necessary Finite Element matrices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    start = time.time()
    Eimg = mpimg.imread(filename)
    if Eimg.ndim == 3:
        Eimg = Eimg[:, :, 0]  # Take first channel if RGB

    # --- Mesh Generation ---
    # Downsample significantly for speed if needed
    if Eimg.shape[0] > target_size:
        AE = resize(Eimg, (target_size, target_size), anti_aliasing=True)
    else:
        AE = Eimg
        
    A = resize(AE, (np.minimum(AE.shape[0], AE.shape[1]), np.minimum(AE.shape[0], AE.shape[1])))
    area = 'qa0.0005'
    height = 30e-3  # mm
    width = A.shape[1] / A.shape[0] * 30e-3  # mm
    verta = height
    vertb = width
    pts2 = np.array([[0, 0], [verta, 0], [0, verta / 2], [verta, vertb], [verta, vertb / 2], [0, vertb]])
    roi = dict(vertices=pts2)
    mesh_raw = triangle.triangulate(roi, area)

    vertices = mesh_raw['vertices']
    triangles = mesh_raw['triangles']
    N = len(vertices)
    
    # Grid data interpolation to map image pixels to mesh nodes
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    x_new = np.linspace(x.min(), x.max(), N)
    y_new = np.linspace(y.min(), y.max(), N)
    X, Y = np.meshgrid(x_new, y_new, indexing='ij')
    
    step = 10
    x0 = np.arange(0, A.shape[0], step)
    y0 = np.arange(0, A.shape[1], step)
    xi, yi = np.meshgrid(x0, y0)
    
    xi = xi / np.max(xi) * 30e-3
    yi = yi / np.max(yi) * 30e-3
    x1 = np.squeeze(np.reshape(xi, (xi.size, 1)))
    y1 = np.squeeze(np.reshape(yi, (yi.size, 1)))
    A1 = np.squeeze(np.reshape(A[::step, ::step], (A[::step, ::step].size, 1)))
    z = griddata((x1, y1), A1, (X, Y), method='linear')
    
    # Custom GridData logic integrated here
    xmin, xmax = y.min(), y.max() # Note the swap in original code usage
    ymin, ymax = x.min(), x.max()
    binsize = (xmax - xmin) / (N - 1)
    xi_g = np.arange(xmin, xmax + binsize, binsize)
    yi_g = np.arange(ymin, ymax + binsize, binsize)
    xi_g, yi_g = np.meshgrid(xi_g, yi_g)
    
    zvect = np.zeros((N, 1))
    nrow, ncol = xi_g.shape
    for row in range(nrow):
        for col in range(ncol):
            xc = xi_g[row, col]
            yc = yi_g[row, col]
            posx = np.abs(y - xc) # Note swap
            posy = np.abs(x - yc) # Note swap
            ibin = np.logical_and(posx < binsize / 2., posy < binsize / 2.)
            ind = np.where(ibin == True)[0]
            if len(ind) > 0:
                zvect[ind] = z[row, col]

    E_ground_truth = np.ravel(zvect).astype('float32') * 1e+5
    
    # --- FEM Matrices Calculation ---
    n_element = len(triangles)
    Ae = np.zeros(n_element)
    for i in range(n_element):
        Ax = vertices[triangles[i, 0], 0]
        Ay = vertices[triangles[i, 0], 1]
        Bx = vertices[triangles[i, 1], 0]
        By = vertices[triangles[i, 1], 1]
        Cx = vertices[triangles[i, 2], 0]
        Cy = vertices[triangles[i, 2], 1]
        Ae[i] = (1 / 2) * np.abs(Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))

    Be = np.zeros((3, 6, len(Ae)))
    for i in range(len(Ae)):
        y23 = vertices[triangles[i, 1], 1] - vertices[triangles[i, 2], 1]
        y31 = vertices[triangles[i, 2], 1] - vertices[triangles[i, 0], 1]
        y12 = vertices[triangles[i, 0], 1] - vertices[triangles[i, 1], 1]
        x32 = vertices[triangles[i, 2], 0] - vertices[triangles[i, 1], 0]
        x13 = vertices[triangles[i, 0], 0] - vertices[triangles[i, 2], 0]
        x21 = vertices[triangles[i, 1], 0] - vertices[triangles[i, 0], 0]
        Be[:, :, i] = (1 / (2 * Ae[i])) * np.array([
            [y23, 0, y31, 0, y12, 0],
            [0, x32, 0, x13, 0, x21],
            [x32, y23, x13, y31, x21, y12]
        ])

    # Global Stiffness Setup
    v = 0.495 # Poisson
    n_node = len(vertices)
    Tens = np.zeros((2 * n_node, 2 * n_node, n_node)).astype('float32')
    KT = np.zeros((2 * n_node, 2 * n_node))
    
    c11 = (1 - v) / ((1 + v) * (1 - 2 * v))
    c12 = v / ((1 + v) * (1 - 2 * v))
    c66 = 1 / (2 * (1 + v))
    C = np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c66]])
    
    rho = 1000
    w_freq = 2 * np.pi * 90
    d = rho * w_freq**2
    kt_base = np.array([
        [2 * d, 0, d, 0, d, 0],
        [0, 2 * d, 0, d, 0, d],
        [d, 0, 2 * d, 0, d, 0],
        [0, d, 0, 2 * d, 0, d],
        [d, 0, d, 0, 2 * d, 0],
        [0, d, 0, d, 0, 2 * d]
    ])

    for i in range(n_element):
        kt = -1 * Ae[i] / 12 * kt_base
        Ke = Ae[i] / 3 * np.dot(np.dot((np.transpose(Be[:, :, i])), C), Be[:, :, i])
        nodes = triangles[i, :]
        a, b, c = nodes[0], nodes[1], nodes[2]
        idx = [2*a, 2*a+1, 2*b, 2*b+1, 2*c, 2*c+1]
        for r in range(6):
            for col in range(6):
                Tens[idx[r], idx[col], [a, b, c]] += Ke[r, col]
                KT[idx[r], idx[col]] += kt[r, col]

    # Boundaries for Forward Problem
    # Re-implement boundaries2 logic
    f = 10
    flag = 1
    inn = triangles
    
    # GK calculation for forward solve requires E
    GK = np.squeeze(Tens @ E_ground_truth) / 3 + KT

    height = 30e-3
    tb = height
    width = 30e-3
    bcy = -0.2
    fxy = np.zeros(2 * n_node)
    
    mesh_b = np.zeros((n_element, 3, 4))
    for i in range(n_element):
        for j in range(3):
            mesh_b[i, j, 0] = inn[i, j]
            mesh_b[i, j, 1:3] = vertices[inn[i, j], :]
            
    w_centre = width / 2
    
    for i in range(n_element):
        for j in range(3):
            if mesh_b[i, j, 2] == 0:  # Bottom
                fxy[(inn[i, j]) * 2 + 1] = 0
                GK[(inn[i, j]) * 2 + 1, :] = 0
                GK[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1] = 1
                if mesh_b[i, j, 1] == w_centre:
                    fxy[(inn[i, j]) * 2] = 0
                    GK[(inn[i, j]) * 2, :] = 0
                    GK[(inn[i, j]) * 2, (inn[i, j]) * 2] = 1
            if mesh_b[i, j, 2] == tb and flag == 1:
                GK[(inn[i, j]) * 2 + 1, :] = 0
                GK[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1] = 1
                fxy[(inn[i, j]) * 2 + 1] = tb * bcy
    
    # Boundary logic for Tensor (used in Inverse)
    Tens_bc = Tens.copy()
    for i in range(n_element):
        for j in range(3):
            if mesh_b[i, j, 2] == 0:
                Tens_bc[(inn[i, j]) * 2 + 1, :, :] = 0
                Tens_bc[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1, :] = 1/10000
                if mesh_b[i, j, 1] == w_centre:
                    Tens_bc[(inn[i, j]) * 2, :, :] = 0
                    Tens_bc[(inn[i, j]) * 2, (inn[i, j]) * 2, :] = 1/10000
            if mesh_b[i, j, 2] == tb and flag == 1:
                Tens_bc[(inn[i, j]) * 2 + 1, :, :] = 0
                Tens_bc[(inn[i, j]) * 2 + 1, (inn[i, j]) * 2 + 1, :] = 1/10000
                
    matTens = np.transpose(Tens_bc, (2, 0, 1))

    print(f"Data Loaded. E shape: {E_ground_truth.shape}, Vertices: {vertices.shape}")
    return E_ground_truth * 1e-5, GK, Tens_bc, KT, matTens, triangles, vertices, fxy

# ==========================================
# 2. Forward Operator
# ==========================================

def forward_operator(x_stiffness, GK_structure, fxy):
    """
    Solves the forward Elasticity problem K(E)u = f to get displacement u.
    Note: x_stiffness should be scaled appropriately (e.g. ~1e5 or normalized).
    If using the refactored load output, x_stiffness might be normalized.
    """
    # In the original code, 'disp1' calculation uses GK modified by boundaries.
    # GK was already constructed with E_ground_truth in load_and_preprocess.
    # However, strictly speaking, a forward operator should take 'x' and produce 'y'.
    
    # Since GK is dependent on E linearly: K(E) = sum(E_i * K_i) + M.
    # But for boundary conditions, rows of GK are modified.
    # To keep it simple and consistent with the provided input code's logic flow:
    # We will use the pre-computed GK which contains the ground truth E and BCs.
    
    # Solve Ku = f
    GK_s = scipy.sparse.csr_matrix(GK_structure)
    disp = scipy.sparse.linalg.spsolve(GK_s, fxy)
    disp = np.around(disp, decimals=10) * 1e+5 # Scale up as in original code
    
    return disp

# ==========================================
# 3. Inversion (Reconstruction)
# ==========================================

def _im2bw(Ig, level):
    S = np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return S

def _vect2im(A, vertices):
    N = len(A)
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    x_new = np.linspace(x.min(), x.max(), N)
    y_new = np.linspace(y.min(), y.max(), N)
    X, Y = np.meshgrid(x_new, y_new)
    A_im = np.around(griddata((x, y), np.ravel(A), (X, Y), method='linear'), decimals=10)
    return A_im

def _nodeneighbor(triangles, N):
    T = [[] for _ in range(N)]
    for elem in triangles:
        for node in elem:
            T[node].extend(elem)
    for i in range(N):
        T[i] = list(set(T[i]))
    T1 = np.zeros((N, 10))
    for i in range(N):
        length = min(len(T[i]), 10)
        T1[i, :length] = T[i][:length]
    return T1

def _ggradient(E, A):
    N = np.size(E)
    grad1 = np.zeros((N, 1))
    for n in range(N):
        S1 = 0
        for k in range(10):
            if A[n, k] != 0:
                neighbor_idx = int(A[n, k])
                if neighbor_idx < N:
                    S1 = S1 + (E[n] - E[neighbor_idx])**2
        grad1[n] = np.sqrt(S1)
    if np.max(grad1) != 0:
        result1 = grad1 / np.max(np.ceil(grad1))
    else:
        result1 = grad1
    return np.ravel(result1)

def run_inversion(disp_measured, E_initial_guess, matTens, Tens, KT, triangles, vertices, fxy, params):
    """
    Performs the Proximal Optimization to reconstruct Stiffness (E) from Displacement (disp_measured).
    """
    start = time.time()
    
    # Unpack params
    tau1 = params.get('tau1', 0.94)
    tau2 = params.get('tau2', 0.06)
    tau3 = params.get('tau3', 0.01)
    step = params.get('step', 0.16 * 0.6)
    maxit = params.get('maxit', 20)
    outer_itr = params.get('outer_itr', 1)
    
    um = disp_measured
    N = matTens.shape[0] # Number of nodes (E is defined on nodes in this formulation)
    
    # Construct Operator D for ||y - D x||
    # Dm = matTens @ um -> (N, 2N)
    Dm = matTens @ um
    D2 = (Dm.T) / 3 
    
    # Target vector
    ft = fxy - KT @ um * 1e-5
    
    # Initial Solution
    sol = E_initial_guess
    
    # TV Operator Helper
    T1 = _nodeneighbor(triangles, N)
    g = lambda x: _ggradient(x, T1)
    
    # Covariance for Mahalanobis distance term
    # We need a GK0 estimate to build Gamma. 
    # The original code updates GK0 inside the outer loop.
    
    yy2 = np.ravel(ft)
    N_cov = np.zeros((2 * N, 2 * N))
    for j in np.arange(0, 2 * N, 2):
        N_cov[j, j] = 3
        N_cov[j + 1, j + 1] = 1
        
    w = np.ones((2 * N, 1))
    w[:102 * 2, :] = 0.99
    
    for j in range(outer_itr):
        GK0 = (np.squeeze(Tens @ sol / 3) + 1e-5 * KT)
        
        # Build Gamma
        gamma = GK0 @ N_cov @ GK0.T
        gamma += np.eye(gamma.shape[1]) * 1e-5
        gamma_inv = np.linalg.inv(gamma)
        
        # Define Functions for PyUnlocBox
        
        # 1. Quadratic Data Fidelity with Covariance
        f8 = functions.func(lambda_=tau1)
        f8._eval = lambda x: 0.5 * (yy2 - D2 @ x).T @ gamma_inv @ (yy2 - D2 @ x) * 1e-4
        f8._grad = lambda x: -1 * D2.T @ gamma_inv @ (yy2 - D2 @ x) * 1e-4
        
        # 2. TV / Smoothness
        f3 = functions.norm_l1(A=g, At=None, dim=1, y=np.zeros(N), lambda_=tau2)
        
        # 3. Box Constraint [0, 0.7] (Normalized Stiffness)
        f2 = functions.func()
        f2._eval = lambda x: 0
        f2._prox = lambda x, T: np.clip(x, 0, 0.7)
        
        # Solver
        solver = solvers.generalized_forward_backward(step=step)
        
        # Solve
        print(f"  Starting Optimization Loop {j+1}/{outer_itr}...")
        ret = solvers.solve([f8, f3, f2], np.copy(sol), solver, rtol=1e-15, maxit=maxit, verbosity='LOW')
        sol = ret['sol']
        
        if np.any(np.isnan(sol)) or np.mean(sol) > 1:
            print("  Warning: Solver diverged or produced invalid values.")
            break
            
    print("Inversion time: %.4f seconds" % (time.time() - start))
    return sol

# ==========================================
# 4. Evaluation
# ==========================================

def evaluate_results(E_true, E_recon, vertices):
    """
    Calculates CNR and RMSE, and plots the results.
    """
    # Calculate Metrics
    thresh = 0.8 * np.max(E_recon)
    idxe = np.where(E_recon > thresh)
    idxb = np.where(E_recon < thresh)
    
    EE = E_recon[idxe[0]]
    BB = E_recon[idxb[0]]
    
    if len(EE) == 0 or len(BB) == 0:
        cnr = 0
    else:
        cnr = 10 * np.log10(2 * (np.mean(EE) - np.mean(BB))**2 / (np.var(EE) + np.var(BB)))
        
    rms = np.sqrt(np.mean(np.abs(2 * (E_recon - E_true) / (E_true + E_recon + 1e-9))**2))
    
    print("\nEvaluation Results:")
    print(f"CNR: {cnr:.2f} dB")
    print(f"RMSE: {rms:.4f}")

    # Plotting
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_new, y_new)
    
    E_true_im = griddata((x, y), E_true, (X, Y), method='linear')
    E_recon_im = griddata((x, y), E_recon, (X, Y), method='linear')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(E_true_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    plt.title("Ground Truth Stiffness")
    
    plt.subplot(1, 2, 2)
    plt.imshow(E_recon_im, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    plt.title(f"Reconstructed (CNR={cnr:.1f}dB)")
    
    plt.savefig('mre_refactored_result.png')
    print("Result saved to mre_refactored_result.png")
    
    return cnr, rms

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # --- Configuration ---
    filename = 'inputs/E_clean630.png'
    
    # Handle missing file for demo purposes
    if not os.path.exists(filename):
        # Create a dummy image if file doesn't exist
        print(f"Warning: {filename} not found. Creating synthetic data.")
        os.makedirs('inputs', exist_ok=True)
        synthetic_img = np.zeros((100, 100))
        synthetic_img[30:70, 30:70] = 1.0 # Inclusion
        plt.imsave(filename, synthetic_img, cmap='gray')

    # 1. Load Data
    print("Loading Data...")
    E_true, GK, Tens, KT, matTens, triangles, vertices, fxy = load_and_preprocess_data(filename)
    
    # 2. Forward Model
    print("Running Forward Model...")
    disp_clean = forward_operator(E_true, GK, fxy)
    
    # Add Noise
    ymeas_noise_coef = 1e-3
    print(f"Adding Noise (coef={ymeas_noise_coef})...")
    
    dispx = disp_clean[0::2]
    dispy = disp_clean[1::2]
    N = len(dispx)
    np.random.seed(1)
    
    xmeas_noise_coef = 1.7 * ymeas_noise_coef
    stdx = xmeas_noise_coef * np.abs(dispx)
    umx = dispx + np.random.normal(0, stdx, N)
    umx = umx + np.random.normal(0, (xmeas_noise_coef * (np.max(umx) - np.min(umx))), N)
    
    stdy = ymeas_noise_coef * np.abs(dispy)
    umy = dispy + np.random.normal(0, stdy, N)
    umy = umy + np.random.normal(0, (ymeas_noise_coef * (np.max(umy) - np.min(umy))), N)
    
    um = np.zeros(2 * N)
    um[0::2] = umx
    um[1::2] = umy
    
    # 3. Initialization Logic
    # -----------------------
    back_init = 0.1
    scale = 0.34
    varibx0 = ymeas_noise_coef * 2
    stdx0 = varibx0 * np.abs(E_true)
    E_interp = _vect2im(np.ravel(E_true) + np.random.normal(0, stdx0, len(E_true)), vertices)
    I1 = _im2bw(np.abs(E_interp), 0.22)
    x1_im = E_interp - back_init * np.ones(E_interp.shape) + scale * I1
    
    # Map back to nodes
    x_init = np.zeros(len(E_true))
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    idx_x = np.floor(x_norm * (x1_im.shape[1] - 1)).astype(int)
    idx_y = np.floor(y_norm * (x1_im.shape[0] - 1)).astype(int)
    
    # griddata/image indexing check: image usually (row, col) -> (y, x)
    for j in range(len(E_true)):
        x_init[j] = x1_im[idx_y[j], idx_x[j]]
    
    # 4. Inversion
    print("Running Inversion...")
    inv_params = {
        'maxit': 20, 
        'outer_itr': 1, 
        'step': 0.16 * 0.6,
        'tau1': 0.94,
        'tau2': 0.06
    }
    
    E_recon = run_inversion(um, x_init, matTens, Tens, KT, triangles, vertices, fxy, inv_params)
    
    # 5. Evaluation
    evaluate_results(E_true, E_recon, vertices)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")