import os

import numpy as np

import SimpleITK as sitk

from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

import warnings

warnings.filterwarnings("ignore")

def reorient_to_ras(image):
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        image.SetDirection((1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0))
    elif array.ndim == 3:
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
    return image

def downsample_image(image_array, time_index=None, t_limit=100, z_limit=25, y_limit=250, x_limit=250):
    def generate_indices(original_size, target_size):
        if target_size >= original_size:
            return np.arange(original_size)
        return np.linspace(0, original_size - 1, target_size, dtype=int)
    
    def calculate_spatial_dims(x_size, y_size, x_limit, y_limit):
        if x_size <= x_limit and y_size <= y_limit:
            return x_size, y_size
        x_scale = x_limit / x_size if x_size > x_limit else 1.0
        y_scale = y_limit / y_size if y_size > y_limit else 1.0
        scale_factor = min(x_scale, y_scale)
        return int(x_size * scale_factor), int(y_size * scale_factor)
    
    if image_array.ndim == 4:
        t_size, z_size, y_size, x_size = image_array.shape
        new_t = min(t_size, t_limit)
        new_z = min(z_size, z_limit)
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit)

        indices = [
            generate_indices(t_size, new_t),
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]
        downsampled = image_array[np.ix_(*indices)]
        if time_index is not None:
            new_time = [time_index[i] for i in indices[0]]
            return downsampled, new_time
        return downsampled

    elif image_array.ndim == 3:
        z_size, y_size, x_size = image_array.shape
        new_z = min(z_size, z_limit)
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit)
        indices = [
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]
        return image_array[np.ix_(*indices)]
    
    return image_array

def generate_mask_ctp_helper(volume, bone_thresh=300, fm_thresh=400):
    img_sitk = sitk.GetImageFromArray(volume)
    img_sitk = sitk.DiscreteGaussian(img_sitk)
    
    # Bone segmentation
    bone_mask = img_sitk > bone_thresh
    comps = sitk.ConnectedComponent(bone_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(comps)
    
    if stats.GetNumberOfLabels() == 0:
        return np.zeros_like(volume)

    # Find skull (largest bone component usually, or max physical size)
    labels = stats.GetLabels()
    # Heuristic: pick label with largest physical size
    sizes = [(l, stats.GetPhysicalSize(l)) for l in labels]
    seed_label = max(sizes, key=lambda x: x[1])[0]
    seed_pt = img_sitk.TransformPhysicalPointToIndex(stats.GetCentroid(seed_label))
    
    # Fast Marching
    grad = sitk.GradientMagnitudeRecursiveGaussian(img_sitk, sigma=0.5)
    speed = sitk.BoundedReciprocal(grad)
    
    fm = sitk.FastMarchingBaseImageFilter()
    fm.SetTrialPoints([seed_pt])
    fm.SetStoppingValue(fm_thresh)
    fm_out = fm.Execute(speed)
    
    mask_sitk = fm_out < fm_thresh
    return sitk.GetArrayFromImage(mask_sitk)

def load_and_preprocess_data(image_path, mask_path=None, scan_interval=1.0, gaussian_sigma=1.0):
    """
    Loads NIfTI data, reorients, downsamples, smooths, and ensures a mask exists.
    """
    print(f"Loading data from {image_path}...")
    
    # Load 4D Image
    sitk_img = sitk.ReadImage(image_path)
    sitk_img = reorient_to_ras(sitk_img)
    arr_4d = sitk.GetArrayFromImage(sitk_img)
    
    # Create time index
    t_len = arr_4d.shape[0]
    time_index = [i * scan_interval for i in range(t_len)]
    
    # Downsample
    arr_4d, time_index = downsample_image(arr_4d, time_index)
    
    # Gaussian Smoothing
    # Process as list of 3D volumes
    volume_list = []
    for t in range(arr_4d.shape[0]):
        vol = arr_4d[t]
        vol_smooth = gaussian_filter(vol, sigma=gaussian_sigma)
        volume_list.append(vol_smooth)
        
    # Mask Handling
    mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask from {mask_path}...")
        m_img = sitk.ReadImage(mask_path)
        m_img = reorient_to_ras(m_img)
        m_arr = sitk.GetArrayFromImage(m_img)
        mask = downsample_image(m_arr)
    else:
        print("Generating mask from data...")
        # Use the first volume or mean volume for masking
        ref_vol = volume_list[0]
        mask = generate_mask_ctp_helper(ref_vol)
        
    return np.stack(volume_list, axis=0), np.array(time_index), mask
