import numpy as np
import scipy.io as sio
import scipy.linalg as spl
from scipy import signal
import odl
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# =============================================================================
# Helper Functions (Internal Logic)
# =============================================================================

def estimate_cov(I1, I2):
    """
    Estimate the covariance of I1 and I2 using a Laplacian-like filter.
    This emphasizes high-frequency noise.
    """
    assert I1.shape == I2.shape
    H, W = I1.shape
    
    # Laplacian kernel to filter out smooth signal and keep noise
    M = np.array([[1, -2, 1],
                  [-2, 4., -2],
                  [1, -2, 1]])
    
    # Convolve and compute scalar product
    sigma = np.sum(signal.convolve2d(I1, M) * signal.convolve2d(I2, M))
    sigma /= (W * H - 1)
    
    # Normalization factor (empirical or derived from M)
    return sigma / 36.0

def cov_matrix(data):
    """
    Estimate the covariance matrix from data (stack of images/sinograms).
    data: (N, H, W)
    Returns: (N, N) covariance matrix.
    """
    n = len(data)
    cov_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            cov_mat[i, j] = estimate_cov(data[i], data[j])
    return cov_mat

# =============================================================================
# Core Functional Components
# =============================================================================

def load_and_preprocess_data(reco_space):
    """
    Loads generated data and creates a 2-material problem with correlated noise.
    
    Args:
        reco_space: ODL discretization space.
        
    Returns:
        data_noisy: (2, n_angles, det_size) numpy array.
        gt_images: (2, H, W) numpy array or None.
        geometry: ODL geometry object.
    """
    data_path = 'data/material_proj_data'
    phantom_path = 'raw_phantom'
    
    if not os.path.exists(data_path):
        # Fallback simulation if files don't exist, to ensure code runs for the prompt requirement
        print(f"Warning: {data_path} not found. Synthesizing random phantom data.")
        n_angles = 360
        det_size = 512
        angle_partition = odl.uniform_partition(0.0, 2.0 * np.pi, n_angles)
        detector_partition = odl.uniform_partition(-det_size / 2.0, det_size / 2.0, det_size)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                            src_radius=500, det_radius=500)
        
        # Create dummy ground truth
        shepp = odl.phantom.shepp_logan(reco_space, modified=True)
        gt_images = np.array([shepp.asarray(), 0.5 * shepp.asarray()])
        
        # Create forward op for dummy data
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
        proj1 = ray_trafo(gt_images[0]).asarray()
        proj2 = ray_trafo(gt_images[1]).asarray()
        
        data_clean = np.array([proj1, proj2])
        
    else:
        mat_data = sio.loadmat(data_path)
        phantom_data = sio.loadmat(phantom_path) if os.path.exists(phantom_path) else None
        
        # 1. Prepare Projections (Sinograms)
        sino_bone = mat_data['bone']
        sino_denser = mat_data['denser_sphere']
        sino_brain = mat_data['brain']
        sino_csf = mat_data['csf']
        sino_blood = mat_data['blood']
        sino_eye = mat_data['eye']
        sino_less_dense = mat_data['less_dense_sphere']
        
        proj_mat1 = sino_bone + sino_denser
        proj_mat2 = sino_brain + sino_csf + sino_blood + sino_eye + sino_less_dense
        data_clean = np.array([proj_mat1, proj_mat2])
        
        # 2. Prepare Ground Truth Images
        if phantom_data:
            img_bone = phantom_data['bone']
            img_denser = phantom_data['denser_sphere']
            img_brain = phantom_data['brain']
            img_csf = phantom_data['csf']
            img_blood = phantom_data['blood']
            img_eye = phantom_data['eye']
            img_less_dense = phantom_data['less_dense_sphere']
            
            gt_mat1 = img_bone + img_denser
            gt_mat2 = img_brain + img_csf + img_blood + img_eye + img_less_dense
            
            gt_mat1_res = resize(gt_mat1, reco_space.shape, anti_aliasing=True)
            gt_mat2_res = resize(gt_mat2, reco_space.shape, anti_aliasing=True)
            gt_images = np.array([gt_mat1_res, gt_mat2_res])
        else:
            gt_images = None
        
        # 3. Setup Geometry
        n_angles = proj_mat1.shape[0]
        det_size = proj_mat1.shape[1]
        angle_partition = odl.uniform_partition(0.0, 2.0 * np.pi, n_angles)
        detector_partition = odl.uniform_partition(-det_size / 2.0, det_size / 2.0, det_size)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                            src_radius=500, det_radius=500)

    # 4. Add Correlated Noise
    scale = 0.05 * np.max(data_clean)
    cov_true = np.array([[1.0, -0.8], [-0.8, 0.8]]) * (scale**2)
    
    n_angles_curr = data_clean.shape[1]
    det_size_curr = data_clean.shape[2]
    
    noise_flat = np.random.multivariate_normal([0, 0], cov_true, size=(n_angles_curr * det_size_curr))
    noise = noise_flat.reshape(n_angles_curr, det_size_curr, 2)
    noise = np.moveaxis(noise, -1, 0)
    
    data_noisy = data_clean + noise
    
    return data_noisy, gt_images, geometry


def forward_operator(x, space, geometry):
    """
    Applies the Ray Transform to the input x.
    Note: Ideally x is an ODL element, but if it is numpy, we wrap it.
    
    Args:
        x: Input volume (ODL element or numpy array).
        space: ODL reconstruction space.
        geometry: ODL geometry.
        
    Returns:
        y_pred: The forward projection (numpy array).
    """
    # Setup Ray Transform
    if odl.tomo.ASTRA_CUDA_AVAILABLE:
        impl = 'astra_cuda'
    else:
        impl = 'astra_cpu'
        
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)
    
    # Diagonal Operator to apply RayTransform to both material channels
    A = odl.DiagonalOperator(ray_trafo, 2)
    
    if isinstance(x, np.ndarray):
        x_odl = A.domain.element(x)
        return A(x_odl).asarray()
    else:
        # Assume x is an ODL element compatible with A
        return A(x).asarray()


def run_inversion(data, space, geometry):
    """
    Sets up and solves the inverse problem using Douglas-Rachford Primal-Dual.
    
    Args:
        data: Noisy sinogram data (2, angles, detectors).
        space: ODL reconstruction space.
        geometry: ODL geometry.
        
    Returns:
        result: Reconstructed volume (2, H, W) numpy array.
    """
    # 1. Forward Operator Setup
    if odl.tomo.ASTRA_CUDA_AVAILABLE:
        impl = 'astra_cuda'
    else:
        impl = 'astra_cpu'
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)
    A = odl.DiagonalOperator(ray_trafo, 2)
    
    # 2. Noise Estimation & Whitening
    cov_est = cov_matrix(data)
    w_mat = spl.fractional_matrix_power(cov_est, -0.5)
    
    # Whitening Operator
    I_range = odl.IdentityOperator(ray_trafo.range)
    W = odl.ProductSpaceOperator(np.multiply(w_mat, I_range))
    
    # Whitened Forward Operator and Data
    op = W * A
    rhs = W(data)
    
    # Data Matching Functional: || W(Ax - b) ||^2
    data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)
    
    # 3. Regularization (Joint Prior: Nuclear Norm of Gradient)
    grad = odl.Gradient(space)
    L = odl.DiagonalOperator(grad, 2)
    lambda_reg = 0.15 
    regularizer = lambda_reg * odl.solvers.NuclearNorm(L.range)
    
    # 4. Solver Setup
    # Initial guess: FBP
    fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.7)
    x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])
    
    # Constraint: Non-negativity
    f_func = odl.solvers.IndicatorBox(A.domain, 0, np.inf)
    
    g_funcs = [data_discrepancy, regularizer]
    lin_ops = [op, L]
    
    # Step size estimation
    op_norm = odl.power_method_opnorm(op)
    grad_norm = odl.power_method_opnorm(grad)
    tau = 1.0
    sigma = (1.0/op_norm**2, 1.0/grad_norm**2)
    
    # Solve
    niter = 20
    callback = odl.solvers.CallbackPrintIteration()
    odl.solvers.douglas_rachford_pd(x, f_func, g_funcs, lin_ops, niter, 
                                    tau=tau, sigma=sigma, callback=callback)
    
    return x.asarray()


def evaluate_results(reconstruction, gt_images):
    """
    Computes PSNR and SSIM if ground truth is available.
    
    Args:
        reconstruction: (2, H, W) numpy array.
        gt_images: (2, H, W) numpy array or None.
    """
    if gt_images is None:
        print("Ground truth not available for quantitative evaluation.")
        return

    # Clip negative values
    res_images = np.maximum(reconstruction, 0)
    
    metrics = []
    material_names = ["Bone/Calcium", "Soft Tissue/Water"]
    
    print("\n=== Evaluation ===")
    for i in range(2):
        gt = gt_images[i]
        rec = res_images[i]
        
        # Normalize both to [0,1] for fair comparison (handles unit mismatch)
        gt_min, gt_max = np.min(gt), np.max(gt)
        rec_min, rec_max = np.min(rec), np.max(rec)
        denom_gt = gt_max - gt_min if gt_max != gt_min else 1.0
        denom_rec = rec_max - rec_min if rec_max != rec_min else 1.0
        gt = (gt - gt_min) / denom_gt
        rec = (rec - rec_min) / denom_rec
        
        # Dynamic range for PSNR (now both in [0,1])
        dmax = 1.0
        
        p = psnr(gt, rec, data_range=dmax)
        s = ssim(gt, rec, data_range=dmax)
        metrics.append((p, s))
        
        print(f"Material {i+1} ({material_names[i]}): PSNR = {p:.2f} dB, SSIM = {s:.4f}")
        
    avg_psnr = np.mean([m[0] for m in metrics])
    avg_ssim = np.mean([m[1] for m in metrics])
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    import os

    # Define I/O directory
    io_dir = './io'
    os.makedirs(io_dir, exist_ok=True)

    # Define Reconstruction Space
    space = odl.uniform_discr([-129, -129], [129, 129], [512, 512])

    # 1. Load Data
    print("Step 1: Loading Data...")
    data, gt_images, geometry = load_and_preprocess_data(space)

    # >>> SAVE INPUT <<<
    np.save(os.path.join(io_dir, 'input_noisy_projections.npy'), data)
    print(f"Input saved to {io_dir}/")
    
    # 2. Forward Operator (Demonstration / Check)
    # Ideally, we verify the operator works, but run_inversion builds its own operators internally
    # to handle the whitening logic cleanly. However, we can test projection here.
    print("Step 2: verifying forward operator...")
    dummy_proj = forward_operator(np.zeros((2, 512, 512)), space, geometry)
    assert dummy_proj.shape == data.shape
    
    # 3. Run Inversion
    print("Step 3: Running Inversion...")
    reconstruction = run_inversion(data, space, geometry)

    # >>> SAVE OUTPUT <<<
    np.save(os.path.join(io_dir, 'output_reconstruction.npy'), reconstruction)
    print(f"Output saved to {io_dir}/")
    
    # 4. Evaluate
    print("Step 4: Evaluating Results...")
    evaluate_results(reconstruction, gt_images)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")