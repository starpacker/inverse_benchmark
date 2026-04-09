import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.fftpack import next_fast_len


def load_image(fp, dtype="float32"):
    """Load an image file and return as numpy array."""
    from PIL import Image
    img = Image.open(fp)
    img_array = np.array(img).astype(dtype)
    return img_array


def normalize_image(img):
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img


def downsample_image(img, factor):
    """Downsample image by a given factor using simple slicing."""
    if factor == 1:
        return img
    if len(img.shape) == 2:
        return img[::factor, ::factor]
    elif len(img.shape) == 3:
        return img[::factor, ::factor, :]
    else:
        return img


def prepare_4d_array(img):
    """Ensure image is 4D: (depth, height, width, channels)."""
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if len(img.shape) == 3:
        img = img[np.newaxis, :, :, :]
    return img


def rfft2_convolve_setup(psf, dtype=np.float32):
    """Set up FFT-based convolution parameters."""
    psf = psf.astype(dtype)
    psf_shape = np.array(psf.shape)
    
    padded_shape = 2 * psf_shape[-3:-1] - 1
    padded_shape = np.array([next_fast_len(int(i)) for i in padded_shape])
    padded_shape = list(np.r_[psf_shape[-4], padded_shape, psf.shape[-1]].astype(int))
    
    start_idx = ((np.array(padded_shape[-3:-1]) - psf_shape[-3:-1]) // 2).astype(int)
    end_idx = (start_idx + psf_shape[-3:-1]).astype(int)
    
    return {
        "psf": psf,
        "psf_shape": psf_shape,
        "padded_shape": padded_shape,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "dtype": dtype
    }


def pad_array(v, setup):
    """Pad array to padded_shape."""
    padded_shape = setup["padded_shape"]
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    
    if len(v.shape) == 5:
        batch_size = v.shape[0]
        shape = [batch_size] + padded_shape
    elif len(v.shape) == 4:
        shape = padded_shape
    else:
        raise ValueError("Expected 4D or 5D tensor")
    
    vpad = np.zeros(shape).astype(v.dtype)
    vpad[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :] = v
    return vpad


def crop_array(x, setup):
    """Crop array from padded_shape to original shape."""
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    return x[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :]


def compute_psf_fft(setup):
    """Compute FFT of padded PSF."""
    psf = setup["psf"]
    padded_psf = pad_array(psf, setup)
    H = fft.rfft2(padded_psf, axes=(-3, -2), norm="ortho")
    return H


def convolve_fft(x, H, setup, pad=True):
    """Perform convolution using FFT."""
    padded_shape = setup["padded_shape"]
    
    if pad:
        x_padded = pad_array(x, setup)
    else:
        x_padded = x
    
    conv_output = fft.rfft2(x_padded, axes=(-3, -2), norm="ortho") * H
    conv_output = fft.ifftshift(
        fft.irfft2(conv_output, axes=(-3, -2), s=padded_shape[-3:-1], norm="ortho"),
        axes=(-3, -2),
    )
    
    if pad:
        conv_output = crop_array(conv_output, setup)
    
    return conv_output.real.astype(setup["dtype"])


def deconvolve_fft(y, H_adj, setup, pad=True):
    """Perform adjoint convolution (correlation) using FFT."""
    padded_shape = setup["padded_shape"]
    
    if pad:
        y_padded = pad_array(y, setup)
    else:
        y_padded = y
    
    deconv_output = fft.rfft2(y_padded, axes=(-3, -2), norm="ortho") * H_adj
    deconv_output = fft.ifftshift(
        fft.irfft2(deconv_output, axes=(-3, -2), s=padded_shape[-3:-1], norm="ortho"),
        axes=(-3, -2),
    )
    
    if pad:
        deconv_output = crop_array(deconv_output, setup)
    
    return deconv_output.real.astype(setup["dtype"])


def power_method_lipschitz(setup, H, H_adj, max_iter=20):
    """Estimate Lipschitz constant using power method."""
    psf_shape = setup["psf_shape"]
    dtype = setup["dtype"]
    
    x = np.random.randn(*psf_shape).astype(dtype)
    x /= np.linalg.norm(x)
    
    for _ in range(max_iter):
        conv_x = convolve_fft(x, H, setup, pad=True)
        x = deconvolve_fft(conv_x, H_adj, setup, pad=True)
        norm_val = np.linalg.norm(x)
        if norm_val > 0:
            x /= norm_val
    
    return norm_val


def prox_nonneg(x):
    """Proximal operator for non-negativity constraint."""
    return np.maximum(x, 0)


# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(psf_path, data_path, downsample=4):
    """
    Load PSF and measurement data, preprocess them.
    
    Args:
        psf_path: Path to PSF image file
        data_path: Path to measurement data file
        downsample: Downsampling factor
    
    Returns:
        Dictionary containing preprocessed PSF and measurement data
    """
    print(f"Loading data from {data_path}...")
    print(f"Loading PSF from {psf_path}...")
    
    psf_raw = load_image(psf_path, dtype="float32")
    data_raw = load_image(data_path, dtype="float32")
    
    psf_ds = downsample_image(psf_raw, downsample)
    data_ds = downsample_image(data_raw, downsample)
    
    psf_norm = normalize_image(psf_ds)
    data_norm = normalize_image(data_ds)
    
    psf = prepare_4d_array(psf_norm)
    data = prepare_4d_array(data_norm)
    
    print(f"Data shape: {data.shape}")
    print(f"PSF shape: {psf.shape}")
    
    return {"psf": psf, "data": data}


# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_operator(image_est, psf):
    """
    Apply the forward model: convolve image estimate with PSF.
    
    Args:
        image_est: Estimated image (4D array)
        psf: Point spread function (4D array)
    
    Returns:
        Predicted measurement (convolution result)
    """
    dtype = psf.dtype
    setup = rfft2_convolve_setup(psf, dtype=dtype)
    H = compute_psf_fft(setup)
    
    y_pred = convolve_fft(image_est, H, setup, pad=True)
    
    return y_pred


# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(data_dict, n_iter=50):
    """
    Run APGD (FISTA) reconstruction algorithm.
    
    Args:
        data_dict: Dictionary with 'psf' and 'data' keys
        n_iter: Number of iterations
    
    Returns:
        Reconstructed image as numpy array
    """
    psf = data_dict["psf"]
    measurement = data_dict["data"]
    dtype = psf.dtype
    
    setup = rfft2_convolve_setup(psf, dtype=dtype)
    H = compute_psf_fft(setup)
    H_adj = np.conj(H)
    
    print("Estimating Lipschitz constant...")
    L = power_method_lipschitz(setup, H, H_adj, max_iter=20)
    print(f"Lipschitz constant L = {L:.4e}")
    
    x_k = np.zeros_like(psf)
    y_k = x_k.copy()
    t_k = 1.0
    
    step_size = 1.0 / L if L > 0 else 1.0
    
    print(f"Starting APGD for {n_iter} iterations...")
    start_time = time.time()
    
    for i in range(n_iter):
        if i % 10 == 0:
            print(f"  Iteration {i}/{n_iter}")
        
        Ay_k = convolve_fft(y_k, H, setup, pad=True)
        residual = Ay_k - measurement
        gradient = deconvolve_fft(residual, H_adj, setup, pad=True)
        
        x_k_next_unprox = y_k - step_size * gradient
        
        x_k_next = prox_nonneg(x_k_next_unprox)
        
        t_k_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        y_k = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        
        x_k = x_k_next
        t_k = t_k_next
    
    end_time = time.time()
    print(f"Reconstruction finished in {end_time - start_time:.2f}s")
    
    result = x_k
    if result.shape[0] == 1:
        result = result[0]
    
    return result


# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(reconstruction, output_path="result.png"):
    """
    Evaluate and save reconstruction results.
    
    Args:
        reconstruction: Reconstructed image array
        output_path: Path to save the result image
    """
    print(f"Saving result to {output_path}...")
    
    img = reconstruction.copy()
    
    if len(img.shape) == 4:
        img = img[0]
    
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_display = (img - img_min) / (img_max - img_min)
    else:
        img_display = img
    
    img_display = np.clip(img_display, 0, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    if len(img_display.shape) == 3 and img_display.shape[-1] == 3:
        ax.imshow(img_display)
    elif len(img_display.shape) == 3 and img_display.shape[-1] == 1:
        ax.imshow(img_display[:, :, 0], cmap='gray')
    elif len(img_display.shape) == 2:
        ax.imshow(img_display, cmap='gray')
    else:
        ax.imshow(img_display)
    
    ax.set_title("APGD Reconstruction")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")
    
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Reconstruction min: {reconstruction.min():.6f}")
    print(f"Reconstruction max: {reconstruction.max():.6f}")
    print(f"Reconstruction mean: {reconstruction.mean():.6f}")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    psf_file = "data/psf/tape_rgb.png"
    data_file = "data/raw_data/thumbs_up_rgb.png"
    
    if not os.path.exists(psf_file):
        print(f"Error: {psf_file} not found.")
        exit(1)
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        exit(1)
    
    data_dict = load_and_preprocess_data(psf_file, data_file, downsample=4)
    
    reconstruction = run_inversion(data_dict, n_iter=5)
    
    evaluate_results(reconstruction, "apgd_result_inlined.png")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")