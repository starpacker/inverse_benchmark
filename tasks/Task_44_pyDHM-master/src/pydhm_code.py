import numpy as np
import cv2
import os
from math import pi, sqrt, log10

# =============================================================================
# Helper Functions (Defined before use)
# =============================================================================

def angularSpectrum(field, z, wavelength, dx, dy):
    """
    Function to diffract a complex field using the angular spectrum approximation
    Extracted logic from input code.
    """
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * N)
    dfy = 1 / (dy * M)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    # Transfer function
    # Note: Using np.exp for phase
    phase_term = np.exp(1j * z * 2 * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2)) + 0j))

    tmp = field_spec * phase_term

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out

def PS4(Inp0, Inp1, Inp2, Inp3):
    '''
    Function to recover the phase information of a sample from four DHM in-axis acquisitions holograms
    '''
    inp0 = np.array(Inp0)
    inp1 = np.array(Inp1)
    inp2 = np.array(Inp2)
    inp3 = np.array(Inp3)

    # compensation process
    # U_obj ~ (I3-I1)j + (I2-I0)
    comp_phase = (inp3 - inp1) * 1j + (inp2 - inp0)

    return comp_phase

def calculate_psnr(img1, img2):
    """Calculates PSNR between two normalized images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two images.
    """
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Gaussian kernel for local mean
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def quantize_image(img):
    """Simulates 8-bit quantization."""
    # Assuming the img is the intensity hologram, normalize per set usually, 
    # but here applied locally for simulation logic
    # In the main flow, we will handle max_val properly if available, 
    # or just assume the image range.
    # To keep this function pure, we assume 'img' is the raw intensity.
    # This specific implementation follows the input code logic which normalized 
    # by the max of the set. Since we are inside a helper, we'll do simple 
    # scaling if needed, but the original logic requires context of 4 images.
    # We will implement the quantization logic inside the forward_operator 
    # where all 4 images are available.
    pass 

# =============================================================================
# 1. load_and_preprocess_data
# =============================================================================

def load_and_preprocess_data(image_path, target_shape=None):
    """
    Loads the ground truth amplitude, synthesizes phase, and creates the complex field.
    
    Returns:
        field_input (complex np.array): The complex object field.
        gt_amp (np.array): Ground truth amplitude [0, 1].
        gt_phase (np.array): Ground truth phase.
    """
    print(f"Loading Ground Truth Image: {image_path}")
    
    if not os.path.exists(image_path):
        # Trying a few fallbacks as per original logic if exact path fails, 
        # though the caller should ideally provide a valid path.
        # Here we just raise error if it really doesn't exist to be strict.
        raise FileNotFoundError(f"Error: File not found at {image_path}")

    gt_amp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gt_amp is None:
        raise ValueError("Error: Failed to read image.")
        
    # Resize if target_shape is provided (optional for flexibility)
    if target_shape is not None:
        gt_amp = cv2.resize(gt_amp, (target_shape[1], target_shape[0]))

    # Normalize to [0, 1]
    gt_amp = gt_amp.astype(np.float32) / 255.0
    
    # Generate Synthetic Phase (e.g., spherical phase)
    M, N = gt_amp.shape
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - N/2, y - M/2)
    # A simple phase lens-like structure
    gt_phase = 5 * np.exp(-(X**2 + Y**2) / (2 * (200**2))) 
    
    # Create Complex Object
    field_input = gt_amp * np.exp(1j * gt_phase)
    
    return field_input, gt_amp, gt_phase

# =============================================================================
# 2. forward_operator
# =============================================================================

def forward_operator(field_input, z, wavelength, dx, dy, add_quantization=True):
    """
    Simulates the hologram recording process:
    1. Propagation to hologram plane.
    2. Interference with phase-shifted reference waves.
    3. (Optional) Camera quantization.
    
    Returns:
        I_stack (list of np.array): List containing [I0, I1, I2, I3].
    """
    # Propagate the field to the hologram plane
    hologram_field_complex = angularSpectrum(field_input, z, wavelength, dx, dy)
    
    # Simulate Reference Wave (Plane Wave, Amplitude 1)
    R = 1.0 
    
    # Phase shifts: 0, pi/2, pi, 3pi/2
    I0 = np.abs(hologram_field_complex + R * np.exp(1j * 0))**2
    I1 = np.abs(hologram_field_complex + R * np.exp(1j * pi/2))**2
    I2 = np.abs(hologram_field_complex + R * np.exp(1j * pi))**2
    I3 = np.abs(hologram_field_complex + R * np.exp(1j * 3*pi/2))**2
    
    if add_quantization:
        # Simulate 8-bit Camera Quantization
        # Normalize by the global maximum of the stack to preserve relative intensities
        max_val = np.max([I0.max(), I1.max(), I2.max(), I3.max()])
        
        def quantize_single(img, limit):
            img_norm = img / limit
            img_8bit = np.round(img_norm * 255).astype(np.uint8)
            # Convert back to float scale
            return img_8bit.astype(np.float32) / 255.0 * limit

        I0 = quantize_single(I0, max_val)
        I1 = quantize_single(I1, max_val)
        I2 = quantize_single(I2, max_val)
        I3 = quantize_single(I3, max_val)
        
    return [I0, I1, I2, I3]

# =============================================================================
# 3. run_inversion
# =============================================================================

def run_inversion(I_stack, z, wavelength, dx, dy):
    """
    Performs the reconstruction:
    1. Phase shifting retrieval (PS4).
    2. Back-propagation to object plane.
    
    Returns:
        reconstructed_field (np.complex): The complex object field at z=0.
    """
    I0, I1, I2, I3 = I_stack
    
    # Step 1: Recover Complex Field at Hologram Plane using PS4
    # The PS4 formula returns (I3-I1)j + (I2-I0) which is proportional to 4*R*O
    # Since R=1, we get 4*O. We need to divide by 4 to get true scale.
    recovered_holo_field = PS4(I0, I1, I2, I3) / 4.0
    
    # Step 2: Propagate back to the object plane (-z)
    reconstructed_field = angularSpectrum(recovered_holo_field, -z, wavelength, dx, dy)
    
    return reconstructed_field

# =============================================================================
# 4. evaluate_results
# =============================================================================

def evaluate_results(reconstructed_field, gt_amp, gt_phase):
    """
    Calculates metrics and saves images.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    reconstructed_amplitude = np.abs(reconstructed_field)
    reconstructed_phase = np.angle(reconstructed_field)
    
    # Clip result to valid range [0, 1] for amplitude comparison
    reconstructed_amplitude_clipped = np.clip(reconstructed_amplitude, 0, 1)
    
    psnr_val = calculate_psnr(gt_amp, reconstructed_amplitude_clipped)
    ssim_val = calculate_ssim(gt_amp, reconstructed_amplitude_clipped)
    
    print(f"Amplitude PSNR: {psnr_val:.2f} dB")
    print(f"Amplitude SSIM: {ssim_val:.4f}")
    
    # Save outputs
    cv2.imwrite('output_gt_amp.png', (gt_amp * 255).astype(np.uint8))
    cv2.imwrite('output_reconstruction_amp.png', (reconstructed_amplitude_clipped * 255).astype(np.uint8))
    
    # Visualize Phase (Normalize to 0-255 for visualization)
    gt_phase_norm = ((gt_phase - gt_phase.min()) / (gt_phase.max() - gt_phase.min()) * 255).astype(np.uint8)
    rec_phase_norm = ((reconstructed_phase - reconstructed_phase.min()) / (reconstructed_phase.max() - reconstructed_phase.min()) * 255).astype(np.uint8)
    
    cv2.imwrite('output_gt_phase.png', gt_phase_norm)
    cv2.imwrite('output_reconstruction_phase.png', rec_phase_norm)
    
    print("Saved output images: output_gt_amp.png, output_reconstruction_amp.png, output_gt_phase.png, output_reconstruction_phase.png")
    
    return {"PSNR": psnr_val, "SSIM": ssim_val}

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    wavelength = 633e-9  # 633 nm
    dx = 5e-6            # 5 um pixel pitch
    dy = 5e-6
    z = 0.05             # 5 cm propagation distance
    
    # Define Image Path logic (local to script execution)
    image_rel_path = 'data/numericalPropagation samples/horse.bmp'
    possible_paths = [
        image_rel_path,
        os.path.join(os.getcwd(), 'pyDHM-master', image_rel_path),
        '/data/yjh/pyDHM-master/data/numericalPropagation samples/horse.bmp'
    ]
    
    image_path = None
    for p in possible_paths:
        if os.path.exists(p):
            image_path = p
            break
            
    if image_path is None:
        # Generate a dummy image if file not found to strictly allow code to run
        # This prevents crash in environments without the specific file
        print("Warning: 'horse.bmp' not found. Generating synthetic rectangle.")
        dummy_img = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(dummy_img, (150, 150), (350, 350), 255, -1)
        cv2.imwrite('temp_dummy_horse.bmp', dummy_img)
        image_path = 'temp_dummy_horse.bmp'

    # 1. Load Data
    field_input, gt_amp, gt_phase = load_and_preprocess_data(image_path)
    
    # 2. Forward Operator
    I_stack = forward_operator(field_input, z, wavelength, dx, dy, add_quantization=True)
    
    # Save one hologram for visualization check
    I0_vis = I_stack[0]
    cv2.imwrite('output_simulated_hologram_I0.png', (255 * I0_vis / np.max(I0_vis)).astype(np.uint8))

    # 3. Run Inversion
    reconstructed_field = run_inversion(I_stack, z, wavelength, dx, dy)
    
    # 4. Evaluate Results
    evaluate_results(reconstructed_field, gt_amp, gt_phase)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")