import os
import sys
import types
import math
import logging
import bz2
import pickle
import numpy as np
import scipy.fftpack
from numpy.fft import rfft2, irfft2
from scipy.interpolate import griddata
import scipy.ndimage
from skimage import io, metrics, transform, util
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functools import reduce

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger('semiblind')

# ==========================================
# Device Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Helper Functions (from data_utils.py)
# ==========================================

def scale(v):
    '''
    Normalize a 2D matrix with a maximum of 1 per pixel
    '''
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    return out * (1/np.max(np.abs(out)))

def normalize(v):
    '''
    Normalize a 2D matrix with a sum of 1
    '''
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm

def gaussian_kernel(size, fwhmx=3, fwhmy=3, center=None):
    """ Make a square gaussian kernel. """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return normalize(np.exp(-4 * np.log(2) * (((x - x0) ** 2) / fwhmx**2 + ((y - y0) ** 2) / fwhmy**2)))

def unpad(img, npad):
    ''' Revert the np.pad command '''
    return img[npad:-npad, npad:-npad]

# ==========================================
# Model Classes (from model.py)
# ==========================================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn1.track_running_stats = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Hack to allow torch.load to find the model classes
model_module = types.ModuleType('model')
model_module.ResNet = ResNet
model_module.BasicBlock = BasicBlock
model_module.Bottleneck = Bottleneck
sys.modules['model'] = model_module

# ==========================================
# Deconvolution Logic (from deconvolution.py)
# ==========================================

def compute_grid_in_memory(psf_map, input_image):
    """
    Computes interpolation grid coefficients and returns them directly.
    """
    grid_z1 = []
    grid_x, grid_y = np.mgrid[0:input_image.shape[0], 0:input_image.shape[1]]
    xmax = np.linspace(0, input_image.shape[0], psf_map.shape[0])
    ymax = np.linspace(0, input_image.shape[1], psf_map.shape[1])

    total_patches = psf_map.shape[0]*psf_map.shape[1]
    
    for i in range(total_patches):
        # log.info('Compute interpolation for patch:{}/{}'.format(i, total_patches))
        points = []
        values = []
        for x in xmax:
            for y in ymax:
                points.append(np.asarray([x, y]))
                values.append(0.0)

        values[i] = 1.0

        points = np.asarray(points)
        values = np.asarray(values)

        grid_z1.append(griddata(points, values, (grid_x, grid_y), method='linear', rescale=True))
    
    return grid_z1

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0
    return c

def _centered(arr, newshape):
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def divergence(F):
    return reduce(np.add, np.gradient(F))

def rl_deconv_all(img_list, psf_list, iterations=10, lbd=0.2):
    """
    Spatially-Variant Richardson-lucy deconvolution with Total Variation regularization
    """
    min_value = []
    processed_img_list = []
    
    # Pad images
    pad_width = np.max(psf_list[0].shape)
    for img in img_list:
        img_padded = np.pad(img, pad_width, mode='reflect')
        min_val = np.min(img)
        img_padded = img_padded - min_val
        min_value.append(min_val)
        processed_img_list.append(img_padded)
        
    img_list = processed_img_list
    size = np.array(img_list[0].shape) # Already padded size
    # The original code added psf size to img size for fft size, but here img is already padded?
    # Original: img_list[img_idx] = np.pad(...)
    # size = np.array(np.array(img_list[0].shape) + np.array(psf_list[0].shape)) - 1
    # Actually, standard fft convolution usually pads to size + psf_size - 1.
    # The code calculates fsize based on this.
    
    # Let's trust the original logic's sizing
    # But wait, original code:
    # img_list[img_idx] = np.pad(...)
    # size = np.array(np.array(img_list[0].shape) + np.array(psf_list[0].shape)) - 1
    # This implies full convolution size.
    
    fft_shape_calc = np.array(img_list[0].shape) + np.array(psf_list[0].shape) - 1
    fsize = [scipy.fftpack.next_fast_len(int(d)) for d in fft_shape_calc]
    fslice = tuple([slice(0, int(sz)) for sz in fft_shape_calc])

    latent_estimate = [img.copy() for img in img_list]
    error_estimate = [img.copy() for img in img_list]

    psf_f = []
    psf_flipped_f = []
    for img_idx, img in enumerate(latent_estimate):
        psf_f.append(rfft2(psf_list[img_idx], fsize))
        _psf_flipped = np.flip(psf_list[img_idx], axis=0)
        _psf_flipped = np.flip(_psf_flipped, axis=1)
        psf_flipped_f.append(rfft2(_psf_flipped, fsize))

    for i in range(iterations):
        log.info('RL TV Iter {}/{}'.format(i+1, iterations))
        regularization = np.ones(img_list[0].shape)

        for img_idx, img in enumerate(latent_estimate):
            # Forward projection
            estimate_convolved = irfft2(np.multiply(psf_f[img_idx], rfft2(latent_estimate[img_idx], fsize)))[fslice].real
            estimate_convolved = _centered(estimate_convolved, img.shape)
            
            relative_blur = div0(img_list[img_idx], estimate_convolved)
            
            # Back projection
            error_estimate[img_idx] = irfft2(np.multiply(psf_flipped_f[img_idx], rfft2(relative_blur, fsize)), fsize)[fslice].real
            error_estimate[img_idx] = _centered(error_estimate[img_idx], img.shape)
            
            # TV Regularization
            div_val = divergence(latent_estimate[img_idx] / (np.linalg.norm(latent_estimate[img_idx], ord=1) + 1e-10))
            regularization += 1.0 - (lbd * div_val)
            
            latent_estimate[img_idx] = np.multiply(latent_estimate[img_idx], error_estimate[img_idx])

        for img_idx, img in enumerate(img_list):
            latent_estimate[img_idx] = np.divide(latent_estimate[img_idx], regularization/float(len(img_list)))

    # Unpad and restore min value
    for img_idx, img in enumerate(latent_estimate):
        latent_estimate[img_idx] += min_value[img_idx]
        latent_estimate[img_idx] = unpad(latent_estimate[img_idx], pad_width)

    return np.sum(latent_estimate, axis=0)

# ==========================================
# Test/Demo Logic (from test.py)
# ==========================================

def live_moving_window(im, model, step=64):
    """
    Computes the focus paramter map on the input image using a moving window
    """
    size = 128
    num_classes = 2
    x = size
    y = size
    # Crop to multiple of size
    im = im[0:im.shape[0]//size * size, 0:im.shape[1]//size * size]
    weight_image = np.zeros((im.shape[0], im.shape[1], num_classes))

    tile_dataset = []
    
    # Sliding window extraction
    # Re-implementing the while loops cleanly
    for x_end in range(size, im.shape[0] + 1, step):
        for y_end in range(size, im.shape[1] + 1, step):
            a = im[x_end - size:x_end, y_end - size:y_end]
            a = scale(a)
            tile_dataset.append(a[:])
            weight_image[x_end - size:x_end, y_end - size:y_end] += 1.0

    if not tile_dataset:
        return np.zeros((im.shape[0], im.shape[1], num_classes)), np.zeros((0, num_classes))

    tile_dataset = np.asarray(tile_dataset)
    tile_dataset = np.reshape(tile_dataset, (tile_dataset.shape[0], 1, size, size))
    
    # Batch processing
    max_size = tile_dataset.shape[0]
    batch_size = 8
    it = 0
    output_npy = np.zeros((tile_dataset.shape[0], num_classes))
    input_tensor = torch.FloatTensor(tile_dataset)

    model.eval()
    with torch.no_grad():
        while max_size > 0:
            num_batch = min(batch_size, max_size)
            batch_input = input_tensor.narrow(0, it, num_batch).to(device)
            out = model(batch_input)
            output_npy[it:it+num_batch] = out.data.cpu().numpy()
            it += num_batch
            max_size -= num_batch

    output = np.zeros((im.shape[0], im.shape[1], output_npy.shape[1]))
    
    i = 0
    for x_end in range(size, im.shape[0] + 1, step):
        for y_end in range(size, im.shape[1] + 1, step):
            output[x_end - size:x_end, y_end - size:y_end] += output_npy[i, :]
            i += 1

    output = output / weight_image
    return output, output_npy

# ==========================================
# Forward Model Implementation
# ==========================================

def apply_spatially_variant_blur(image, psf_grid, grid_weights):
    """
    Simulates spatially variant blur.
    image: 2D numpy array
    psf_grid: list of PSFs
    grid_weights: list of weight maps (same length as psf_grid)
    """
    blurred_image = np.zeros_like(image)
    
    # Pad image for convolution
    pad_width = np.max(psf_grid[0].shape)
    img_padded = np.pad(image, pad_width, mode='reflect')
    
    fft_shape_calc = np.array(img_padded.shape) + np.array(psf_grid[0].shape) - 1
    fsize = [scipy.fftpack.next_fast_len(int(d)) for d in fft_shape_calc]
    fslice = tuple([slice(0, int(sz)) for sz in fft_shape_calc])
    
    img_f = rfft2(img_padded, fsize)
    
    for i, psf in enumerate(psf_grid):
        psf_f = rfft2(psf, fsize)
        convolved = irfft2(np.multiply(psf_f, img_f), fsize)[fslice].real
        convolved = _centered(convolved, img_padded.shape)
        convolved = unpad(convolved, pad_width)
        
        blurred_image += convolved * grid_weights[i]
        
    return blurred_image

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("Initializing Semi-Blind Deconvolution Demo...")
    
    # 1. Load Data
    img_path = 'data/fly.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)
        
    original_img = io.imread(img_path).astype(np.float32) / 255.0
    # Use a crop to speed up and match test.py logic
    # The image is RGB or Gray? data_utils.grayloader uses Image.open, usually RGB unless converted.
    # But code treats it as 2D. Let's check if it has channels.
    if len(original_img.shape) == 3:
        original_img = np.mean(original_img, axis=2)
        
    # Crop center 512x512 or similar to be safe
    h, w = original_img.shape
    crop_size = 512
    cx, cy = h//2, w//2
    # Ensure crop is within bounds
    start_x = max(0, cx - crop_size//2)
    start_y = max(0, cy - crop_size//2)
    end_x = min(h, start_x + crop_size)
    end_y = min(w, start_y + crop_size)
    
    gt_img = original_img[start_x:end_x, start_y:end_y]
    
    # Ensure dimensions are multiples of 64 for the sliding window
    h_new = (gt_img.shape[0] // 64) * 64
    w_new = (gt_img.shape[1] // 64) * 64
    gt_img = gt_img[:h_new, :w_new]
    
    print(f"Processing Image Size: {gt_img.shape}")
    
    # 2. Forward Model: Generate Spatially Variant Blur
    print("Generating Spatially Variant Blur...")
    
    # Define a simple grid of PSFs (e.g. 2x2 grid)
    rows, cols = 2, 2
    psf_map_shape = (rows, cols)
    
    # Generate random or fixed FWHM for each grid point
    # Let's use fixed values to be reproducible
    # Top-left: small blur, Bottom-right: large blur
    fwhm_map_x = np.linspace(1.5, 5.0, rows)
    fwhm_map_y = np.linspace(1.5, 5.0, cols)
    
    psf_list = []
    grid_params = []
    
    # We need a list of PSFs corresponding to the grid points
    # compute_grid expects psf_map.shape to determine how many points
    # It interpolates from these points.
    
    # Let's create a list of PSFs for the grid points
    for r in range(rows):
        for c in range(cols):
            fx = fwhm_map_x[r]
            fy = fwhm_map_y[c]
            psf = gaussian_kernel(31, fx, fy) # Kernel size 31
            psf_list.append(psf)
            grid_params.append((fx, fy))
            
    # Compute interpolation weights
    print("Computing grid weights...")
    grid_weights = compute_grid_in_memory(np.zeros(psf_map_shape), gt_img)
    
    # Apply blur
    blurred_img = apply_spatially_variant_blur(gt_img, psf_list, grid_weights)
    
    # Add some noise
    noise_sigma = 0.001
    blurred_img += np.random.normal(0, noise_sigma, blurred_img.shape)
    
    # Save blurred image
    io.imsave('synthetic_blur.png', util.img_as_ubyte(np.clip(blurred_img, 0, 1)))
    
    # 3. Inverse Model: Reconstruction
    print("Starting Reconstruction...")
    
    # Load Model
    model_path = 'models/model_999.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading PSF Estimation Model from {model_path}...")
    try:
        cnn_model = torch.load(model_path, map_location=device, weights_only=False)
        # Patch for old PyTorch checkpoint compatibility
        for m in cnn_model.modules():
            if isinstance(m, nn.AvgPool2d):
                if not hasattr(m, 'divisor_override'):
                    m.divisor_override = None
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback if class definitions mismatch somehow (should be fixed by Mock)
        sys.exit(1)
        
    cnn_model.to(device)
    
    # Estimate PSF Map
    print("Estimating Local PSFs...")
    step = 64
    # live_moving_window expects image with max 1 (scale function does this inside, but we passed normalized img)
    # We should pass the blurred image.
    
    # Downscale for estimation (as done in test.py)
    # test.py: downscaled = first_img[128:-128, 128:-128] ... scale(downscale_local_mean(...))
    # We will use the blurred image directly, maybe downscaled if it's too clean?
    # The paper says it works on patches.
    
    estimated_map, _ = live_moving_window(blurred_img, cnn_model, step=step)
    
    # Process estimated map (Median filter + Downscale) - mimicking deconvolution_demo
    # The output of live_moving_window is (H, W, 2)
    
    # Downsample the map to a reasonable grid size for deconvolution
    # In test.py, they downsample the map by step?
    # test.py: img_downsampled = img[::step, ::step]
    
    map_downsampled = estimated_map[::step, ::step]
    
    # Median filter on the parameters
    output_filtered = [scipy.ndimage.median_filter(map_downsampled[:,:,i], size=(2,2), mode='reflect') for i in range(2)]
    
    # Flatten map to get list of PSFs
    # We can use the downsampled map directly as the grid points
    
    rec_psf_list = []
    
    # The shape of output_filtered is (grid_h, grid_w)
    grid_h, grid_w = output_filtered[0].shape
    print(f"Reconstruction Grid Size: {grid_h}x{grid_w}")
    
    for i in range(grid_h * grid_w):
        # Convert flat index to 2D
        r, c = i // grid_w, i % grid_w
        fx = output_filtered[0][r, c]
        fy = output_filtered[1][r, c]
        
        # Clip FWHM to reasonable values
        fx = max(0.1, fx)
        fy = max(0.1, fy)
        
        psf = gaussian_kernel(31, fx, fy)
        rec_psf_list.append(psf)
        
    rec_psf_list = np.asarray(rec_psf_list)
    
    # Compute reconstruction weights
    print("Computing reconstruction grid weights...")
    # We need to pass a shape tuple or array to compute_grid_in_memory representing the grid structure
    # compute_grid_in_memory uses psf_map.shape. Here we have (grid_h, grid_w)
    rec_grid_weights = compute_grid_in_memory(np.zeros((grid_h, grid_w)), blurred_img)
    
    # Prepare masked images for RL
    image_masked_list = []
    for i in range(len(rec_psf_list)):
        image_masked_list.append(np.multiply(rec_grid_weights[i], blurred_img))
        
    # Run Deconvolution
    print("Running Richardson-Lucy Deconvolution...")
    deconvolved_img = rl_deconv_all(image_masked_list, rec_psf_list, iterations=15, lbd=0.05)
    
    # Clip result
    deconvolved_img = np.clip(deconvolved_img, 0, 1)
    
    # 4. Evaluation
    print("Evaluating Results...")
    
    # Calculate PSNR and SSIM
    psnr_val = metrics.peak_signal_noise_ratio(gt_img, deconvolved_img, data_range=1.0)
    ssim_val = metrics.structural_similarity(gt_img, deconvolved_img, data_range=1.0)
    
    print(f"==========================================")
    print(f"Evaluation Metrics:")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"==========================================")
    
    io.imsave('restored.png', util.img_as_ubyte(deconvolved_img))
    print("Saved 'synthetic_blur.png' and 'restored.png'.")
