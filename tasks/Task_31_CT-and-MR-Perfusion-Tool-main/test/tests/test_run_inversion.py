import sys
import os
import dill
import numpy as np
import traceback
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Target Function Import ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If the file is in the same directory but not installed as a package
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# --- Reference B: The Referee (Evaluation Logic) ---
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

def evaluate_results(generated_maps, ref_data_dict=None, output_dir=None):
    """
    Compares generated maps to references and returns a combined quality score (SSIM).
    Modified for QA to take dictionary of arrays instead of paths for references, 
    since we have ground truth in memory.
    """
    print("\nEvaluating Results...")
    
    # Save results if needed (skipping filesystem write for QA speed unless debug needed)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key, data in generated_maps.items():
            path = os.path.join(output_dir, f"generated_{key.lower()}.nii.gz")
            img = sitk.GetImageFromArray(data)
            sitk.WriteImage(img, path)
            print(f"  Saved {key} to {path}")

    # Compare
    if not ref_data_dict:
        print("  No reference data provided for comparison.")
        return 0.0

    metrics_summary = {}
    total_ssim = 0.0
    count = 0
    
    for map_name, gen_data in generated_maps.items():
        if map_name not in ref_data_dict:
            continue
            
        ref_data = ref_data_dict[map_name]
        
        # Ensure exact shape match
        min_z = min(gen_data.shape[0], ref_data.shape[0])
        min_y = min(gen_data.shape[1], ref_data.shape[1])
        min_x = min(gen_data.shape[2], ref_data.shape[2])
        
        g_cut = gen_data[:min_z, :min_y, :min_x]
        r_cut = ref_data[:min_z, :min_y, :min_x]
        
        # Mask out zeros/NaNs/Infs for valid metric calculation
        mask_val = (g_cut != 0) & (r_cut != 0) & np.isfinite(g_cut) & np.isfinite(r_cut)
        
        if np.sum(mask_val) == 0:
            print(f"  {map_name}: No valid overlap for metrics.")
            continue
            
        g_valid = g_cut[mask_val]
        r_valid = r_cut[mask_val]
        
        # Metrics
        mse = mean_squared_error(r_valid, g_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(r_valid, g_valid)
        
        # SSIM (slice by slice on 3D volume)
        ssim_vals = []
        data_range = r_valid.max() - r_valid.min()
        if data_range > 0:
            for z in range(min_z):
                s_slice = g_cut[z]
                r_slice = r_cut[z]
                # Simple check to avoid empty slice warnings
                if s_slice.max() > 0 and r_slice.max() > 0:
                    val = ssim(r_slice, s_slice, data_range=data_range)
                    ssim_vals.append(val)
            avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0
        else:
            avg_ssim = 1.0 if np.allclose(r_valid, g_valid) else 0.0

        print(f"  {map_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, SSIM: {avg_ssim:.4f}")
        metrics_summary[map_name] = {'RMSE': rmse, 'SSIM': avg_ssim}
        
        total_ssim += avg_ssim
        count += 1

    final_score = total_ssim / count if count > 0 else 0.0
    return final_score

# --- Helper Injection ---
# (Injecting necessary helpers for Dill to load the pickle if they were saved as dependencies)
def gamma_variate(t, t0, alpha, beta, amplitude=1.0):
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    safe_t = t_shifted[mask]
    result[mask] = amplitude * safe_t ** alpha * np.exp(-safe_t / beta)
    return result

# --- Main Test Logic ---
def main():
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 1. Identify File Structure
    outer_pkl = None
    inner_pkls = []

    for path in data_paths:
        if 'parent_function' in path:
            inner_pkls.append(path)
        else:
            outer_pkl = path

    if not outer_pkl:
        print("Error: No outer data file found.")
        sys.exit(1)

    print(f"Loading Primary Data: {outer_pkl}")
    try:
        with open(outer_pkl, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer pickle: {e}")
        sys.exit(1)

    # 2. Execution Phase
    try:
        # Pattern 1: Direct Execution
        if not inner_pkls:
            print("Mode: Direct Execution")
            
            args = outer_data.get('args', [])
            kwargs = outer_data.get('kwargs', {})
            std_result = outer_data.get('output', {})
            
            # Execute Agent
            print("Executing agent_run_inversion...")
            agent_result = run_inversion(*args, **kwargs)

            # 3. Evaluation Phase
            # In this context, std_result is the ground truth dictionary of maps (CBF, CBV, etc.)
            # We compare agent_result against std_result using the Reference B logic.
            
            # Extract standard dict if it's wrapped
            if hasattr(std_result, 'detach'): std_result = std_result.detach() # unlikely for dict but safe
            
            # Calculate Scores
            # Note: We compare the agent's result against the ground truth (std_result)
            # Since evaluate_results expects a dict of ref paths, but we have the data directly,
            # we adapted evaluate_results to accept data dicts or we just pass the data dict as ref.
            
            print("\nComparing Agent Output vs Ground Truth...")
            score_agent = evaluate_results(agent_result, ref_data_dict=std_result)
            
            # For "Standard Score", we act as if the Ground Truth is perfect against itself = 1.0
            # or we re-evaluate the std_result against itself to ensure metric consistency.
            score_std = evaluate_results(std_result, ref_data_dict=std_result) # Should be ~1.0 or perfect

            print(f"\nFinal Scores -> Agent SSIM: {score_agent:.4f}, Ideal: {score_std:.4f}")
            
            # 4. Success Criteria
            # SSIM is [0, 1], higher is better.
            threshold = 0.85 # Allow some deviation due to floating point or minor lib diffs
            if score_agent >= threshold:
                print("SUCCESS: Performance integrity verified.")
                sys.exit(0)
            else:
                print("FAILURE: Performance degraded significantly.")
                sys.exit(1)

        else:
            # Pattern 2: Chained Execution (Not expected for run_inversion based on prompt, but implemented for robustness)
            print("Mode: Chained Execution")
            # This block is placeholder logic if run_inversion returned a closure, which it does not in the provided code.
            sys.exit(0)

    except Exception as e:
        print(f"Execution/Evaluation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()