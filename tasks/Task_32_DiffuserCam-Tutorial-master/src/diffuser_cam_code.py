import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize as sk_resize
import os

# --- Configuration ---
CONFIG = {
    'f': 0.25,          # Downsampling factor
    'mu1': 1.0,         # Strong data fidelity
    'mu2': 0.05,        # Moderate TV
    'mu3': 1.0,         # Strong Non-negativity
    'tau': 0.0000001,   # Almost zero TV for dense image
    'iters': 20,       # Many iterations 200
    'disp_pic': 10
}

# --- Helper Functions ---

def min_max_scale(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

def resize(img, factor):
    """
    Downsample image by a factor of 1/2^k.
    Simple box filter downsampling to avoid aliasing.
    """
    num = int(-np.log2(factor))
    for i in range(num):
        # Crop to even size if necessary
        h, w = img.shape[:2]
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        img = img[:h_even, :w_even]
        
        img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
    return img

def load_image(path, as_gray=True):
    img = Image.open(path)
    if as_gray:
        img = img.convert('L') # Convert to grayscale
    img = np.array(img, dtype='float32')
    return img

def normalize(img):
    return img / np.linalg.norm(img.ravel())

# --- ADMM Core Logic ---

def SoftThresh(x, tau):
    # numpy automatically applies functions to each element of the array
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def Psi(v):
    # Forward finite difference
    return np.stack((np.roll(v,1,axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)

def PsiT(U):
    # Adjoint of forward finite difference
    diff1 = np.roll(U[...,0],-1,axis=0) - U[...,0]
    diff2 = np.roll(U[...,1],-1,axis=1) - U[...,1]
    return diff1 + diff2

def C(M, full_size, sensor_size):
    # Crop operator: crops central part of size sensor_size from full_size
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return M[top:bottom,left:right]

def CT(b, full_size, sensor_size):
    # Transpose of Crop (Zero Pad): pads sensor_size to full_size
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    # Handle odd dimensions if necessary (though tutorial assumes power of 2 usually)
    # Ensure output is full_size
    pad_top = v_pad
    pad_bottom = full_size[0] - sensor_size[0] - v_pad
    pad_left = h_pad
    pad_right = full_size[1] - sensor_size[1] - h_pad
    
    return np.pad(b, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=(0,0))

def M(vk, H_fft):
    # Convolution operator in freq domain
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk))*H_fft)))

def MT(x, H_fft):
    # Adjoint of convolution
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def U_update(eta, image_est, tau, mu2):
    return SoftThresh(Psi(image_est) + eta/mu2, tau/mu2)

def X_update(xi, image_est, H_fft, sensor_reading, X_divmat, mu1, full_size, sensor_size):
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading, full_size, sensor_size))

def W_update(rho, image_est, mu3):
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft, mu1, mu2, mu3):
    return (mu3*w - rho)+PsiT(mu2*u - eta) + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat, mu1, mu2, mu3):
    freq_space_result = R_divmat*fft.fft2( fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft, mu1, mu2, mu3)) )
    return np.real(fft.fftshift(fft.ifft2(freq_space_result)))

def xi_update(xi, V, H_fft, X, mu1):
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U, mu2):
    return eta + mu2*(Psi(V) - U)

def rho_update(rho, V, W, mu3):
    return rho + mu3*(V - W)

# --- Precomputations ---

def precompute_X_divmat(sensor_size, full_size, mu1): 
    """Only call this function once!"""
    return 1./(CT(np.ones(sensor_size), full_size, sensor_size) + mu1)

def precompute_PsiTPsi(full_size):
    PsiTPsi = np.zeros(full_size)
    PsiTPsi[0,0] = 4
    PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
    PsiTPsi = fft.fft2(PsiTPsi)
    return PsiTPsi

def precompute_R_divmat(H_fft, PsiTPsi, mu1, mu2, mu3): 
    """Only call this function once!"""
    MTM_component = mu1*(np.abs(np.conj(H_fft)*H_fft))
    PsiTPsi_component = mu2*np.abs(PsiTPsi)
    id_component = mu3
    return 1./(MTM_component + PsiTPsi_component + id_component)

def precompute_H_fft(psf, full_size, sensor_size):
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))

def init_Matrices(H_fft, full_size):
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)

    xi = np.zeros_like(M(V,H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)
    return X,U,V,W,xi,eta,rho

def ADMM_Step(X,U,V,W,xi,eta,rho, precomputed, params):
    H_fft, data, X_divmat, R_divmat, full_size, sensor_size = precomputed
    mu1 = params['mu1']
    mu2 = params['mu2']
    mu3 = params['mu3']
    tau = params['tau']
    
    U = U_update(eta, V, tau, mu2)
    X = X_update(xi, V, H_fft, data, X_divmat, mu1, full_size, sensor_size)
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat, mu1, mu2, mu3)
    W = W_update(rho, V, mu3)
    xi = xi_update(xi, V, H_fft, X, mu1)
    eta = eta_update(eta, V, U, mu2)
    rho = rho_update(rho, V, W, mu3)
    
    return X,U,V,W,xi,eta,rho

def runADMM(psf, data, params):
    sensor_size = np.array(psf.shape)
    full_size = 2*sensor_size
    
    H_fft = precompute_H_fft(psf, full_size, sensor_size)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft, full_size)
    X_divmat = precompute_X_divmat(sensor_size, full_size, params['mu1'])
    PsiTPsi = precompute_PsiTPsi(full_size)
    R_divmat = precompute_R_divmat(H_fft, PsiTPsi, params['mu1'], params['mu2'], params['mu3'])
    
    print(f"Starting ADMM for {params['iters']} iterations...")
    for i in range(params['iters']):
        X,U,V,W,xi,eta,rho = ADMM_Step(X,U,V,W,xi,eta,rho, 
                                       [H_fft, data, X_divmat, R_divmat, full_size, sensor_size], 
                                       params)
        if i % 5 == 0:
            print(f"Iteration {i}/{params['iters']}")
            
    image = C(V, full_size, sensor_size)
    image[image<0] = 0
    return image

# --- Main Execution ---

if __name__ == "__main__":
    print("Running DiffuserCam Reconstruction Demo...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    psf_path = os.path.join(base_dir, "tutorial", "psf_sample.tif")
    # Use dog_rgb.jpg as Ground Truth for simulation
    gt_path = os.path.join(base_dir, "test_images", "dog_rgb.jpg")
    
    if not os.path.exists(psf_path) or not os.path.exists(gt_path):
        print(f"Error: Files not found.\nPSF: {psf_path}\nGT: {gt_path}")
        exit(1)
        
    # 1. Load Data
    print("Loading images...")
    psf_raw = load_image(psf_path)
    
    # Use Real Data from Tutorial
    data_path = os.path.join(base_dir, "tutorial", "rawdata_hand_sample.tif")
    if not os.path.exists(data_path):
        print(f"Error: Real data file not found: {data_path}")
        exit(1)
    data_raw = load_image(data_path)
    
    # 2. Preprocess
    # Resize PSF
    f = CONFIG['f']
    psf = resize(psf_raw, f)
    
    # Background subtraction for PSF (as in tutorial)
    bg = np.mean(psf[5:15,5:15]) 
    psf -= bg
    
    # Resize Data
    data = resize(data_raw, f)
    data -= bg # Subtract background from data too as per tutorial
    
    # Normalize
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())
    
    sensor_size = np.array(psf.shape)
    
    print(f"PSF Shape: {psf.shape}")
    print(f"Data Shape: {data.shape}")
    
    # 3. Forward Model (Skipped for Real Data)
    # But we need precomputations for Inverse
    
    # 4. Reconstruct (Inverse Model) - Real Data
    print("Reconstructing (Real Data)...")
    recon_real = runADMM(psf, data, CONFIG)
    
    # --- Synthetic Experiment for Metrics ---
    print("\n" + "="*40)
    print("Running Synthetic Experiment for Metrics (PSNR/SSIM)...")
    
    # Load Ground Truth
    gt_path = os.path.join(base_dir, "test_images", "dog_rgb.jpg")
    if os.path.exists(gt_path):
        gt_raw = load_image(gt_path)
        
        # Resize GT to match sensor size (using skimage to match exactly)
        gt = sk_resize(gt_raw, sensor_size, anti_aliasing=True)
        # Normalize GT
        gt /= np.linalg.norm(gt.ravel())
        
        # Simulate Measurement
        # 1. Pad GT to full size
        full_size = 2 * sensor_size
        gt_padded = CT(gt, full_size, sensor_size)
        
        # 2. Convolve with PSF
        H_fft = precompute_H_fft(psf, full_size, sensor_size)
        meas_sim_full = M(gt_padded, H_fft)
        
        # 3. Crop to sensor size
        meas_sim = C(meas_sim_full, full_size, sensor_size)
        
        # 4. Add Noise (simulate camera noise)
        # Add 1% Gaussian noise to make it realistic
        noise_level = 0.01 * np.max(meas_sim)
        meas_sim += np.random.normal(0, noise_level, meas_sim.shape)
        
        # 5. Normalize (Critical for ADMM parameters to work)
        meas_sim /= np.linalg.norm(meas_sim.ravel())
        
        # Reconstruct Synthetic
        print("Reconstructing (Synthetic Data)...")
        # Increase iterations for synthetic to get better result?
        # But let's stick to config for consistency
        recon_sim = runADMM(psf, meas_sim, CONFIG)
        
        # Calculate Metrics
        def min_max_scale(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        gt_norm = min_max_scale(gt)
        recon_sim_norm = min_max_scale(recon_sim)
        
        # Fix PSNR warning by ensuring range [0, 1]
        val_psnr = psnr(gt_norm, recon_sim_norm, data_range=1.0)
        val_ssim = ssim(gt_norm, recon_sim_norm, data_range=1.0)
        
        print("-" * 30)
        print(f"Synthetic Validation Metrics:")
        print(f"PSNR: {val_psnr:.2f} dB")
        print(f"SSIM: {val_ssim:.4f}")
        print("-" * 30)
        
        # Save Combined Results
        plt.figure(figsize=(12, 10))
        
        # Real Data
        plt.subplot(2, 2, 1)
        plt.imshow(min_max_scale(data), cmap='gray')
        plt.title('Real Sensor Data (Hand)')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(min_max_scale(recon_real), cmap='gray')
        plt.title('Reconstruction (Real Data)')
        plt.axis('off')
        
        # Synthetic Data
        plt.subplot(2, 2, 3)
        plt.imshow(min_max_scale(meas_sim), cmap='gray')
        plt.title('Synthetic Measurement (Dog)')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(recon_sim_norm, cmap='gray')
        plt.title(f'Reconstruction (Synthetic)\nPSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}')
        plt.axis('off')
        
        out_path = os.path.join(base_dir, "reconstruction_result_with_metrics.png")
        plt.savefig(out_path)
        print(f"Result saved to {out_path}")
        
    else:
        print("Warning: Synthetic GT image not found. Skipping metrics.")
        # Fallback to saving just real result
        out_path = os.path.join(base_dir, "reconstruction_result.png")
        plt.imsave(out_path, min_max_scale(recon_real), cmap='gray')
        print(f"Real result saved to {out_path}")
