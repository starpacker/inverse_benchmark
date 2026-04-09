import numpy as np
import scipy.optimize
import scipy.ndimage
import os

# ==============================================================================
# Helper Functions & Classes
# ==============================================================================

class DaxReader:
    """
    Simplified Dax Reader for STORM movies.
    Based on storm_analysis.sa_library.datareader.DaxReader
    """
    def __init__(self, filename):
        self.filename = filename
        self.image_height = None
        self.image_width = None
        self.number_frames = None
        self.bigendian = 0
        
        # Try to read .inf file
        dirname = os.path.dirname(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        inf_filename = os.path.join(dirname, basename + ".inf")
        
        if os.path.exists(inf_filename):
            with open(inf_filename, 'r') as fp:
                for line in fp:
                    if "frame dimensions" in line:
                        dim_str = line.split("=")[1].strip()
                        w, h = dim_str.split("x")
                        self.image_width = int(w)
                        self.image_height = int(h)
                    elif "number of frames" in line:
                        self.number_frames = int(line.split("=")[1].strip())
                    elif "big endian" in line:
                        self.bigendian = 1
        
        if self.image_height is None:
            # Fallback based on file size assuming 256x256
            filesize = os.path.getsize(filename)
            if filesize % (256*256*2) == 0:
                self.image_height = 256
                self.image_width = 256
                self.number_frames = filesize // (256*256*2)
            else:
                raise ValueError("Could not determine DAX dimensions from .inf or file size.")

        self.fileptr = open(filename, "rb")

    def loadAFrame(self, frame_number):
        if frame_number >= self.number_frames:
            raise ValueError("Frame number out of range")
            
        self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
        image_data = np.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
        image_data = np.reshape(image_data, [self.image_height, self.image_width])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data.astype(np.float32)

    def close(self):
        self.fileptr.close()

def symmetric_gaussian_2d(xy, background, height, center_x, center_y, width):
    """
    Explicit mathematical definition of a 2D Symmetric Gaussian.
    f(x,y) = background + height * exp( -2 * ( ((x-cx)/w)^2 + ((y-cy)/w)^2 ) )
    """
    x, y = xy
    g = background + height * np.exp(-2 * (((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2))
    return g.ravel()

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================
def load_and_preprocess_data(file_path, frame_idx, offset, gain):
    """
    Loads a specific frame from a DAX file and applies preprocessing 
    (gain/offset correction).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    reader = DaxReader(file_path)
    try:
        raw_image = reader.loadAFrame(frame_idx)
    finally:
        reader.close()
        
    # Preprocessing
    image = (raw_image - offset) / gain
    image[image < 0] = 0
    
    return image

# ==============================================================================
# 2. Forward Operator
# ==============================================================================
def forward_operator(x_params, image_shape, background_image=None):
    """
    Generates an image from a list of Gaussian parameters.
    x_params: List of [background, height, center_x, center_y, width]
    image_shape: tuple (h, w)
    background_image: Optional 2D array representing global background.
    
    Returns: y_pred (simulated image)
    """
    h, w = image_shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_pred = np.zeros(image_shape)
    
    if background_image is not None:
        y_pred += background_image

    for p in x_params:
        # p: [local_bg, height, cx, cy, wid]
        # We assume local_bg is handled by the global background_image for the full forward model,
        # or we just render the gaussian lobes here.
        # Following the logic of the original code's generate_gaussian_image:
        # Unpack
        _, h_val, cx, cy, wid = p
        
        # Render this gaussian lobe
        g = h_val * np.exp(-2 * (((cx - x_grid) / wid) ** 2 + ((cy - y_grid) / wid) ** 2))
        y_pred += g
        
    return y_pred

# ==============================================================================
# 3. Run Inversion
# ==============================================================================
def run_inversion(image, sigma, background_sigma, threshold_factor):
    """
    Performs peak finding and Gaussian fitting (inversion).
    
    Returns: 
        fitted_params: List of optimized gaussian parameters
        estimated_background: The background image estimated during preprocessing
    """
    # --- Step 1: Preprocessing for Peak Finding ---
    smooth_img = scipy.ndimage.gaussian_filter(image, sigma)
    bg_img = scipy.ndimage.gaussian_filter(image, background_sigma)
    dog_img = smooth_img - bg_img
    
    # Thresholding
    threshold = threshold_factor * np.std(dog_img) + np.mean(dog_img)
    mask = dog_img > threshold
    
    # Local Maxima Detection
    neighborhood_size = 3
    local_max = scipy.ndimage.maximum_filter(dog_img, size=neighborhood_size) == dog_img
    
    # Combined mask
    peaks_mask = local_max & mask
    y_peaks, x_peaks = np.where(peaks_mask)
    
    # --- Step 2: Fitting (Levenberg-Marquardt) ---
    fitted_params = []
    r = 5 
    h, w = image.shape
    
    for px, py in zip(x_peaks, y_peaks):
        if px < r or px >= w - r or py < r or py >= h - r:
            continue
            
        # Crop ROI
        roi = image[py-r:py+r+1, px-r:px+r+1]
        y_roi, x_roi = np.mgrid[py-r:py+r+1, px-r:px+r+1]
        
        # Initial Guess: [background, height, center_x, center_y, width]
        p0 = [np.min(roi), np.max(roi) - np.min(roi), px, py, 2.0]
        
        try:
            error_function = lambda p: symmetric_gaussian_2d((x_roi, y_roi), *p) - roi.ravel()
            p_opt, success = scipy.optimize.leastsq(error_function, p0, maxfev=100)
            
            if success in [1, 2, 3, 4]: 
                bg, height, cx, cy, width = p_opt
                # Sanity checks
                if (height > 0) and (0.5 < width < 10.0) and (abs(cx - px) < 2) and (abs(cy - py) < 2):
                    fitted_params.append(p_opt)
        except Exception:
            continue
            
    return fitted_params, bg_img

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================
def evaluate_results(original, reconstructed):
    """
    Calculates PSNR.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    data_range = np.max(original) - np.min(original)
    if data_range == 0:
        return 0.0
        
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':

    # Define I/O directory
    io_dir = './io'
    os.makedirs(io_dir, exist_ok=True)

    # Define constants
    REPO_PATH = "/data/yjh/storm-analysis-master"
    DAX_FILE = os.path.join(REPO_PATH, "storm_analysis/test/data/test.dax")
    
    PARAMS = {
        'sigma': 1.0,
        'background_sigma': 8.0,
        'threshold': 1.0,
        'camera_offset': 100.0,
        'camera_gain': 1.0,
        'frame_idx': 0
    }

    # 1. Load Data
    try:
        print("Loading data...")
        input_image = load_and_preprocess_data(
            DAX_FILE, 
            PARAMS['frame_idx'], 
            PARAMS['camera_offset'], 
            PARAMS['camera_gain']
        )
        print(f"Data loaded. Shape: {input_image.shape}")

        # 2. Run Inversion
        print("Running inversion...")
        fitted_gaussians, estimated_bg = run_inversion(
            input_image, 
            PARAMS['sigma'], 
            PARAMS['background_sigma'], 
            PARAMS['threshold']
        )
        print(f"Inversion complete. Found {len(fitted_gaussians)} peaks.")

        # 3. Forward Operator (Simulate result for evaluation)
        print("Running forward operator...")
        # The forward model consists of the fitted gaussians + the global estimated background
        reconstructed_image = forward_operator(
            fitted_gaussians, 
            input_image.shape, 
            background_image=estimated_bg
        )

        # >>> SAVE OUTPUT (only one) <<<
        np.save(os.path.join(io_dir, 'output.npy'), reconstructed_image)
        print("saving output")

        # 4. Evaluate
        print("Evaluating results...")
        psnr_score = evaluate_results(input_image, reconstructed_image)
        print(f"PSNR: {psnr_score:.2f} dB")

        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")