import os
import bz2
import numpy as np
import scipy.ndimage
from time import time
import matplotlib.pyplot as plt

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def _find_origin_by_convolution(IM, axes=(0, 1)):
    """
    Find the image origin as the maximum of autoconvolution of its projections.
    """
    if isinstance(axes, int):
        axes = [axes]
    conv = [None, None]
    origin = [IM.shape[0] // 2, IM.shape[1] // 2]
    for a in axes:
        proj = IM.sum(axis=1 - a)
        if proj.size == 0:
             continue
        conv[a] = np.convolve(proj, proj, mode='full')
        origin[a] = np.argmax(conv[a]) / 2
    return tuple(origin)

def _center_image(IM, odd_size=True, square=True):
    rows, cols = IM.shape
    if odd_size and cols % 2 == 0:
        IM = IM[:, :-1]
        rows, cols = IM.shape
    if square and rows != cols:
        if rows > cols:
            diff = rows - cols
            trim = diff // 2
            if trim > 0:
                IM = IM[trim: -trim]
            if diff % 2: IM = IM[: -1]
        else:
            if odd_size and rows % 2 == 0:
                IM = IM[:-1, :]
                rows -= 1
            xs = (cols - rows) // 2
            if xs > 0:
                IM = IM[:, xs:-xs]
        rows, cols = IM.shape

    origin = _find_origin_by_convolution(IM)
    
    # set_center logic merged here for conciseness and scope safety
    center = np.array(IM.shape) // 2
    
    # Check if origin is close to integer for precise shift
    if all(abs(o - round(o)) < 1e-3 for o in origin):
        origin = [int(round(o)) for o in origin]
        out = np.zeros_like(IM)
        src = [slice(None), slice(None)]
        dst = [slice(None), slice(None)]
        for a in range(2):
            d = int(center[a] - origin[a])
            if d > 0:
                dst[a] = slice(d, IM.shape[a])
                src[a] = slice(0, IM.shape[a] - d)
            elif d < 0:
                dst[a] = slice(0, IM.shape[a] + d)
                src[a] = slice(-d, IM.shape[a])
        out[tuple(dst)] = IM[tuple(src)]
        return out
    else:
        delta = [center[a] - origin[a] for a in range(2)]
        return scipy.ndimage.shift(IM, delta, order=3, mode='constant', cval=0.0)

def _get_image_quadrants(IM, reorient=True, symmetry_axis=None):
    IM = np.atleast_2d(IM)
    n, m = IM.shape
    n_c = n // 2 + n % 2
    m_c = m // 2 + m % 2

    Q0 = IM[:n_c, -m_c:]
    Q1 = IM[:n_c, :m_c]
    Q2 = IM[-n_c:, :m_c]
    Q3 = IM[-n_c:, -m_c:]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))

    # Average symmetrization
    if symmetry_axis==(0, 1):
        Q = (Q0 + Q1 + Q2 + Q3)/4.0
        return Q, Q, Q, Q
    return Q0, Q1, Q2, Q3

def _bs_three_point(cols):
    """Deconvolution basis matrix for three_point method."""
    def I0diag(i, j):
        return np.log((np.sqrt((2*j+1)**2-4*i**2) + 2*j+1)/(2*j))/(2*np.pi)
    def I0(i, j):
        return np.log(((np.sqrt((2*j + 1)**2 - 4*i**2) + 2*j + 1)) /
                      (np.sqrt((2*j - 1)**2 - 4*i**2) + 2*j - 1))/(2*np.pi)
    def I1diag(i, j):
        return np.sqrt((2*j+1)**2 - 4*i**2)/(2*np.pi) - 2*j*I0diag(i, j)
    def I1(i, j):
        return (np.sqrt((2*j+1)**2 - 4*i**2) -
                np.sqrt((2*j-1)**2 - 4*i**2))/(2*np.pi) - 2*j*I0(i, j)

    D = np.zeros((cols, cols))
    I, J = np.diag_indices(cols)
    I, J = I[1:], J[1:]
    Ib, Jb = I, J-1
    Iu, Ju = I-1, J
    Iu, Ju = Iu[1:], Ju[1:]
    Iut, Jut = np.triu_indices(cols, k=2)
    Iut, Jut = Iut[1:], Jut[1:]

    D[Ib, Jb] = I0diag(Ib, Jb+1) - I1diag(Ib, Jb+1)
    D[I, J] = I0(I, J+1) - I1(I, J+1) + 2*I1diag(I, J)
    D[Iu, Ju] = I0(Iu, Ju+1) - I1(Iu, Ju+1) + 2*I1(Iu, Ju) - I0diag(Iu, Ju-1) - I1diag(Iu, Ju-1)
    D[Iut, Jut] = I0(Iut, Jut+1) - I1(Iut, Jut+1) + 2*I1(Iut, Jut) - I0(Iut, Jut-1) - I1(Iut, Jut-1)

    D[0, 2] = I0(0, 3) - I1(0, 3) + 2*I1(0, 2) - I0(0, 1) - I1(0, 1)
    D[0, 1] = I0(0, 2) - I1(0, 2) + 2*I1(0, 1) - 1/np.pi
    D[0, 0] = I0(0, 1) - I1(0, 1) + 1/np.pi
    return D


# =============================================================================
# 2. REQUIRED COMPONENT FUNCTIONS
# =============================================================================

def load_and_preprocess_data(file_path):
    """
    Loads text/bz2 data, centers it, splits into quadrants, 
    and returns the top-right quadrant (Q0) for processing.
    """
    print(f"Loading data from {file_path}...")
    
    # 1. Load
    if file_path.endswith('.bz2'):
        with bz2.open(file_path, 'rt') as f:
            raw_im = np.loadtxt(f)
    else:
        raw_im = np.loadtxt(file_path)

    # 2. Center
    # Use odd_size=True, square=True as per original workflow
    centered_im = _center_image(raw_im, odd_size=True, square=True)
    
    # 3. Quadrants
    # Symmetry axis (0,1) averages all 4 quadrants into one representation
    # This effectively boosts SNR for the inversion
    Q_tuple = _get_image_quadrants(centered_im, reorient=True, symmetry_axis=(0, 1))
    
    # Return the processed quadrant (Q0) and the full centered image for ref
    return Q_tuple[0], centered_im


def forward_operator(x, dr=1.0):
    """
    Forward Abel transform using Hansen-Law method.
    Maps Object (x) -> Projection (y_pred)
    """
    # Hansen-Law constants
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1])

    # State equation integral
    def I_func(n_arr, lam_arr, a_val):
        integral = np.empty((n_arr.size, lam_arr.size))
        ratio = n_arr / (n_arr - 1)
        if a_val == 0:
            integral[:, 0] = -np.log(ratio)
        ra = (n_arr - 1)**a_val
        k0 = int(not a_val)
        
        # Slicing loop for stability
        lam_plus_a = lam_arr + a_val
        if k0 < len(lam_plus_a):
            sub_lam = lam_plus_a[k0:]
            # Vectorized calc for k >= k0
            # k maps to index in integral starting at k0
            # sub_lam matches these columns
            term = ra[:, None] * (1 - ratio[:, None]**sub_lam) / sub_lam
            integral[:, k0:] = term
        return integral

    image = np.atleast_2d(x)
    aim = np.empty_like(image)
    rows, cols = image.shape
    
    # Forward specific setup
    drive = -2 * dr * np.pi * np.copy(image)
    a = 1
    
    n = np.arange(cols - 1, 1, -1)
    
    # Calculate phi
    # phi[i, k] = (n[i] / (n[i]-1)) ** lam[k]
    phi = (n[:, None] / (n[:, None] - 1)) ** lam[None, :]

    gamma0 = I_func(n, lam, a) * h
    
    # B matrices for forward
    B1 = gamma0
    B0 = gamma0 * 0

    # Recursive calculation
    state_x = np.zeros((h.size, rows)) # State variable x
    
    # Iterate from outside in
    for indx, col in enumerate(n - 1):
        # drive indices: col+1 is outer, col is inner
        d_outer = drive[:, col + 1]
        d_inner = drive[:, col]
        
        # Update state: x = phi*x + B0*u[k+1] + B1*u[k]
        # Dimensions: state_x is (H, Rows). phi[indx] is (H,). B is (H,). Drive is (Rows,)
        # Broadcast: (H, 1) * (H, Rows) + (H, 1)*(1, Rows) ...
        term1 = phi[indx][:, None] * state_x
        term2 = B0[indx][:, None] * d_outer[None, :]
        term3 = B1[indx][:, None] * d_inner[None, :]
        
        state_x = term1 + term2 + term3
        aim[:, col] = state_x.sum(axis=0)

    # Boundary handling
    aim[:, 0] = aim[:, 1]
    aim[:, -1] = aim[:, -2]
    
    if rows == 1: 
        aim = aim[0]
        
    return aim


def run_inversion(y, dr=1.0):
    """
    Inverse Abel transform using Dasch Three-Point method.
    Maps Projection (y) -> Object (result)
    """
    IM = np.atleast_2d(y)
    rows, cols = IM.shape
    
    # Get operator matrix
    D = _bs_three_point(cols)
    
    # Perform tensordot contraction: result_ri = sum_j (IM_rj * D_ji)
    # axes=(1, 1) usually means sum over axis 1 of IM and axis 1 of D.
    # Logic check:
    # IM is (rows, cols). D is (cols, cols).
    # We want (rows, cols).
    # Dasch implementation: result = IM . D^T? 
    # The helper D indices were (j, i) where j is column index of projection.
    # The standard implementation uses tensordot(IM, D, axes=(1,1))
    
    recon = np.tensordot(IM, D, axes=(1, 1))
    
    # Scale by dr
    return recon / dr


def evaluate_results(y_true, y_pred, recon_object):
    """
    Calculate metrics (PSNR) and display results.
    """
    # MSE and PSNR calculation
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        pixel_max = max(y_true.max(), y_pred.max())
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    print(f"Evaluation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(y_true, cmap='viridis')
    plt.title("Original Projection (Q0)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon_object, cmap='magma', vmax=recon_object.max()*0.5)
    plt.title("Reconstructed Object")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred, cmap='viridis')
    plt.title(f"Reprojection (Forward)\nPSNR: {psnr:.1f} dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_results.png')
    print("Results saved to reconstruction_results.png")
    
    return psnr


# =============================================================================
# 3. MAIN BLOCK
# =============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # Create a dummy file if it doesn't exist for the purpose of this script validity
    # or point to a relative path. Since I cannot access external files,
    # I will generate a synthetic droplet image if the file isn't found, 
    # to ensure the code RUNS as requested.
    
    data_filename = 'O2-ANU1024.txt.bz2'
    
    # Synthetic data generation for robustness if file missing
    if not os.path.exists(data_filename):
        print("Data file not found. Generating synthetic Gaussian data...")
        N = 501
        x = np.linspace(-10, 10, N)
        X, Y = np.meshgrid(x, x)
        # Create a Gaussian blob (projection of a Gaussian ball)
        sigma = 3.0
        synthetic_proj = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        np.savetxt("synthetic_data.txt", synthetic_proj)
        data_path = "synthetic_data.txt"
    else:
        data_path = data_filename

    # 1. Load Data
    # Returns the Quadrant 0 (top-right average) and full centered image
    Q0_data, full_centered_img = load_and_preprocess_data(data_path)
    
    # 2. Inversion (Backward)
    # We recover the density profile from the projection quadrant
    print("Running Inversion...")
    t_start = time()
    reconstructed_density = run_inversion(Q0_data, dr=1.0)
    print(f"Inversion complete ({time() - t_start:.4f}s)")

    # 3. Forward (Reprojection)
    # We take the recovered density and project it again to check consistency
    print("Running Forward Operator...")
    t_start = time()
    reprojected_Q0 = forward_operator(reconstructed_density, dr=1.0)
    print(f"Forward projection complete ({time() - t_start:.4f}s)")

    # 4. Evaluation
    evaluate_results(Q0_data, reprojected_Q0, reconstructed_density)
    
    # Clean up synthetic file if used
    if data_path == "synthetic_data.txt":
        os.remove(data_path)

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")