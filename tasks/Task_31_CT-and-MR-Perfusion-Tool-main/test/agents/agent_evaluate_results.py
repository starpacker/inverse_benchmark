import os

import numpy as np

import SimpleITK as sitk

from skimage.metrics import structural_similarity as ssim

from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def evaluate_results(generated_maps, ref_paths_dict=None, output_dir=None):
    """
    Compares generated maps to references and prints metrics.
    """
    print("\nEvaluating Results...")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key, data in generated_maps.items():
            path = os.path.join(output_dir, f"generated_{key.lower()}.nii.gz")
            img = sitk.GetImageFromArray(data)
            sitk.WriteImage(img, path)
            print(f"  Saved {key} to {path}")

    # Compare
    if not ref_paths_dict:
        print("  No reference paths provided for comparison.")
        return

    metrics_summary = {}
    
    for map_name, gen_data in generated_maps.items():
        ref_key = map_name.lower()
        if ref_key not in ref_paths_dict:
            continue
            
        ref_path = ref_paths_dict[ref_key]
        if not os.path.exists(ref_path):
            print(f"  Reference for {map_name} not found at {ref_path}")
            continue
            
        # Load ref
        ref_img = sitk.ReadImage(ref_path)
        ref_img = reorient_to_ras(ref_img)
        ref_data = sitk.GetArrayFromImage(ref_img)
        ref_data = downsample_image(ref_data) # Ensure shapes match
        
        # Ensure exact shape match (crop if necessary due to downsampling rounding)
        min_z = min(gen_data.shape[0], ref_data.shape[0])
        min_y = min(gen_data.shape[1], ref_data.shape[1])
        min_x = min(gen_data.shape[2], ref_data.shape[2])
        
        g_cut = gen_data[:min_z, :min_y, :min_x]
        r_cut = ref_data[:min_z, :min_y, :min_x]
        
        # Normalize for fair comparison (handle scale diffs)
        # Mask out zeros/NaNs
        mask_val = (g_cut != 0) & (r_cut != 0) & np.isfinite(g_cut) & np.isfinite(r_cut)
        
        if np.sum(mask_val) == 0:
            print(f"  {map_name}: No valid overlap.")
            continue
            
        g_valid = g_cut[mask_val]
        r_valid = r_cut[mask_val]
        
        # Clip extreme outliers using percentiles for robust comparison
        for arr_name, arr in [('gen', g_cut), ('ref', r_cut)]:
            p1, p99 = np.percentile(arr[mask_val], [1, 99])
            if arr_name == 'gen':
                g_cut = np.clip(g_cut, p1, p99)
            else:
                r_cut = np.clip(r_cut, p1, p99)
        g_valid = g_cut[mask_val]
        r_valid = r_cut[mask_val]
        
        # Metrics
        mse = mean_squared_error(r_valid, g_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(r_valid, g_valid)
        
        # PSNR
        data_range = max(r_valid.max() - r_valid.min(), 1e-10)
        if mse > 0:
            psnr_val = 10 * np.log10(data_range**2 / mse)
        else:
            psnr_val = float('inf')
        
        # SSIM (slice by slice)
        ssim_vals = []
        if data_range > 0:
            for z in range(min_z):
                s_slice = g_cut[z]
                r_slice = r_cut[z]
                if s_slice.max() > 0 and r_slice.max() > 0:
                    val = ssim(r_slice, s_slice, data_range=data_range)
                    ssim_vals.append(val)
            avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0
        else:
            avg_ssim = 0.0

        print(f"  {map_name} Comparison -> PSNR: {psnr_val:.2f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, SSIM: {avg_ssim:.4f}")
        metrics_summary[map_name] = {'PSNR': psnr_val, 'RMSE': rmse, 'SSIM': avg_ssim}
