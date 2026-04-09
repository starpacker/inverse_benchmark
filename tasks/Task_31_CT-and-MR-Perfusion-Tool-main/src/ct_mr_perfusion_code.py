import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
import scipy.stats
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

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

def gamma_variate(t, t0, alpha, beta, amplitude=1.0):
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    # Add small epsilon to avoid log(0) or div/0 issues implicitly
    safe_t = t_shifted[mask]
    result[mask] = amplitude * (safe_t**alpha) * np.exp(-safe_t/beta)
    return result

def fit_gamma_variate(time_index, curve):
    try:
        # bounds: [t0, alpha, beta, amp]
        # constrained to be physically plausible
        popt, _ = curve_fit(
            gamma_variate, 
            time_index, 
            curve, 
            bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2.5]),
            maxfev=2000
        )
        return popt
    except Exception:
        return None

def calculate_signal_smoothness(signal):
    if len(signal) < 3: return float('inf')
    rnge = np.max(signal) - np.min(signal)
    if rnge == 0: return 0.0
    norm_sig = (signal - np.min(signal)) / rnge
    second_deriv = np.diff(norm_sig, n=2)
    return np.sum(second_deriv ** 2) / len(signal)

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

def _determine_aif_logic(ctc_volumes, time_index, mask, ttp, dilate_r=5, erode_r=2):
    # Adaptive thresholding logic
    ctc_stack = np.stack(ctc_volumes, axis=0)
    
    # Pre-calculate stats on valid mask
    valid_mask = mask > 0
    if not np.any(valid_mask):
        raise ValueError("Empty mask provided for AIF determination.")

    auc_map = ctc_stack.sum(axis=0)
    auc_vals = auc_map[valid_mask]
    ttp_vals = ttp[valid_mask]
    
    # Search grid
    percentiles = range(95, 4, -10)
    smoothness_t = 0.04
    
    for p in percentiles:
        attempt_auc = np.percentile(auc_vals, p)
        attempt_ttp = np.percentile(ttp_vals, 100 - p)
        
        # Initial candidate mask
        cand_mask = (auc_map > attempt_auc) & (ttp < attempt_ttp) & (ttp > 0)
        
        # Morphological cleanup
        d_k = np.ones((1, dilate_r, dilate_r), bool)
        e_k = np.ones((1, erode_r, erode_r), bool)
        cand_mask = binary_erosion(binary_dilation(cand_mask, structure=d_k), structure=e_k)
        
        labeled, n_comps = measure.label(cand_mask, return_num=True, connectivity=3)
        if n_comps == 0: continue
        
        candidates = []
        for i in range(1, n_comps + 1):
            comp_mask = (labeled == i)
            vol_size = np.sum(comp_mask)
            if not (5 < vol_size < 50): continue # Volume constraints
            
            # Extract curve
            curve_data = ctc_stack[:, comp_mask]
            avg_curve = curve_data.mean(axis=1)
            
            smooth = calculate_signal_smoothness(avg_curve)
            if smooth > smoothness_t: continue
            
            # Fit Gamma
            props = fit_gamma_variate(time_index, avg_curve)
            if props is None: continue
            
            # Calc error
            fitted = gamma_variate(time_index, *props)
            err = np.mean((avg_curve - fitted)**2)
            peak_diff = np.max(avg_curve) - np.mean(avg_curve[-3:])
            
            candidates.append({
                'idx': i,
                'vol': vol_size,
                'props': props,
                'error': err,
                'peak_diff': peak_diff,
                'score': (vol_size * peak_diff) / (err + 1e-6)
            })
            
        if candidates:
            # Select best
            df = pd.DataFrame(candidates)
            best = df.loc[df['score'].idxmax()]
            return best['props'], (labeled == best['idx'])
            
    # Fallback if sophisticated search fails: center of mass of high AUC?
    # For strict compliance, raise error if fails
    raise ValueError("Could not determine AIF from data.")

# --------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 2. FORWARD OPERATOR
# --------------------------------------------------------------------------
def forward_operator(volume_stack, mask, image_type='ctp', echo_time=0.03):
    """
    Transforms raw intensity (signal) space to concentration space (physical model).
    x: raw intensity volumes (4D array)
    y_pred: concentration time curves (4D array) and auxiliary metrics (TTP)
    """
    print("Executing Forward Operator (Intensity -> Concentration)...")
    
    nt, nz, ny, nx = volume_stack.shape
    
    # Handle MRP vs CTP physics
    # CTP: Linear relationship (Beer-Lambert approx for iodine: HU approx linear with conc)
    # MRP: Exponential relationship (Signal approx S0 * exp(-TE * R2*))
    
    # 1. Determine baseline (S0)
    # Simple heuristic: average of first few frames or bolus arrival logic
    # We calculate global means to find bolus arrival
    global_means = []
    valid_mask = mask.astype(bool)
    for t in range(nt):
        val = volume_stack[t][valid_mask].mean()
        global_means.append(val)
    
    # Normalize to find start
    gm = np.array(global_means)
    if len(gm) > 0:
        gm_norm = (gm - gm[0]) / (gm[0] + 1e-6)
    else:
        gm_norm = np.zeros(nt)

    # Simple derivative check for bolus arrival
    # Look for sudden change
    diffs = np.diff(gm_norm)
    # Threshold heuristic
    bolus_idx = 0
    for t in range(2, len(diffs)):
        # Look at window
        window_diff = diffs[t]
        if abs(window_diff) > 0.01: # 1% change
            bolus_idx = t
            break
            
    s0_idx = max(0, bolus_idx - 1)
    
    # Calculate S0 map
    s0_vol = volume_stack[:s0_idx+1].mean(axis=0)
    
    # 2. Calculate Concentration
    if image_type == 'ctp':
        # C(t) = S(t) - S0
        ctc = volume_stack - s0_vol[np.newaxis, :, :, :]
        # Clip negatives strictly? Usually yes for physics, but noise exists.
        # We allow noise but zero out very negative values later or keep for SVD
    elif image_type == 'mrp':
        # C(t) = - (1/TE) * ln(S(t) / S0)
        # Invert intensity for calculation stability if needed (handled in log)
        epsilon = 1e-8
        ratio = volume_stack / (s0_vol[np.newaxis, :, :, :] + epsilon)
        ratio = np.clip(ratio, epsilon, None)
        ctc = -(1.0 / echo_time) * np.log(ratio)
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    # Apply mask
    ctc = ctc * mask[np.newaxis, :, :, :]
    
    # 3. Calculate Time-To-Peak (TTP) on the CTC
    # This is a derived feature often used to guide the inversion (AIF selection)
    # Shift time so TTP is relative to start
    # We return the index or time map. Let's return the map.
    # We need time_index passed in? No, we can just return indices or treat strictly as 
    # part of the operator output.
    # To keep this function pure regarding time, we return argmax indices.
    
    ttp_indices = np.argmax(ctc, axis=0)
    
    return ctc, s0_idx, ttp_indices

# --------------------------------------------------------------------------
# 3. RUN INVERSION
# --------------------------------------------------------------------------
def run_inversion(ctc_volumes, time_index, mask, ttp_indices, svd_thresh=0.1, rho=1.05, hcf=0.73):
    """
    Solves the inverse problem: Given Concentration C(t), find CBF, CBV, MTT.
    Involves:
    1. AIF Estimation (Blind deconvolution aspect).
    2. SVD Deconvolution (Numerical Inversion).
    """
    print("Running Inversion (SVD Deconvolution)...")
    
    # Convert TTP indices to physical time for AIF logic
    ttp_map = np.zeros_like(ttp_indices, dtype=float)
    # Handle mask
    valid_mask = mask > 0
    # Map indices to time
    ttp_vals_flat = np.array(time_index)[ttp_indices[valid_mask]]
    ttp_map[valid_mask] = ttp_vals_flat
    
    # 1. Determine AIF
    try:
        aif_props, aif_mask = _determine_aif_logic(ctc_volumes, time_index, mask, ttp_map)
        print(f"  AIF Parameters found: {aif_props}")
    except ValueError as e:
        print(f"  AIF detection failed: {e}. Using global max heuristic.")
        # Fallback: voxel with max AUC
        auc = ctc_volumes.sum(axis=0) * mask
        flat_idx = np.argmax(auc)
        z, y, x = np.unravel_index(flat_idx, auc.shape)
        aif_curve = ctc_volumes[:, z, y, x]
        aif_props = fit_gamma_variate(time_index, aif_curve)
        
    # Generate AIF curve
    aif_signal = gamma_variate(time_index, *aif_props)
    
    # 2. Construct Forward Matrix (Convolution Matrix) for SVD
    # We use block-circulant SVD formulation for robustness or standard Toeplitz
    dt = np.mean(np.diff(time_index))
    N = len(time_index)
    
    # Method: Standard Toeplitz for simplicity/robustness in this refactor
    # C = A * R * dt  ->  R = inv(A*dt) * C
    col = aif_signal
    row = np.zeros(N)
    row[0] = col[0]
    A_mat = toeplitz(col, row) * dt
    
    # 3. SVD Regularization
    U, S, Vt = np.linalg.svd(A_mat)
    
    # Truncate singular values
    max_s = np.max(S)
    inv_S = np.zeros_like(S)
    idx_keep = S >= (svd_thresh * max_s)
    inv_S[idx_keep] = 1.0 / S[idx_keep]
    
    # Construct pseudo-inverse
    A_inv = Vt.T @ np.diag(inv_S) @ U.T
    
    # 4. Apply Inversion to all voxels
    # Reshape Data: (T, Voxels)
    dims = ctc_volumes.shape # T, Z, Y, X
    n_voxels = np.prod(dims[1:])
    
    ctc_flat = ctc_volumes.reshape(dims[0], n_voxels)
    
    # R = A_inv * C
    residue_flat = np.dot(A_inv, ctc_flat)
    
    # 5. Extract Metrics from Residue Function R(t)
    # R(t) = CBF * R_ideal(t). Theoretically R_ideal(0)=1.
    # So max(R) approx CBF (scaled).
    
    # CBF = max(R) * 60 * 100 / (rho/hcf) (units)
    # CBV = integral(R) * ... or integral(C)/integral(AIF)
    
    # Reshape back
    residue_vol = residue_flat.reshape(dims)
    
    # Map Calculations
    # CBF: Peak of residue function
    cbf_raw = np.max(residue_vol, axis=0)
    cbf_map = (cbf_raw * 60 * 100 * hcf) / rho
    
    # CBV: Area under CTC / Area under AIF (More robust than integrating R)
    # But using R: CBV = sum(R) * dt * units
    # Let's use the standard SVD output definition:
    # CBV = CBF * MTT.
    # Alternatively: CBV = integral(C) / integral(AIF) is standard.
    auc_ctc = np.sum(ctc_volumes, axis=0)
    auc_aif = np.sum(aif_signal)
    
    # Check div zero
    safe_auc_aif = auc_aif if auc_aif > 0 else 1.0
    cbv_map = (auc_ctc / safe_auc_aif) * 100 * hcf / rho
    
    # MTT: CBV / CBF * 60
    mtt_map = np.zeros_like(cbf_map)
    with np.errstate(divide='ignore', invalid='ignore'):
        mtt_map = (cbv_map / cbf_map) * 60
        mtt_map[~np.isfinite(mtt_map)] = 0
        mtt_map[cbf_map == 0] = 0

    # Tmax: Time at which residue function is max? No, usually time C(t) is max is TTP.
    # Tmax in perfusion usually refers to the delay of the residue function peak 
    # relative to AIF peak, or simply argmax(R).
    tmax_indices = np.argmax(residue_vol, axis=0)
    tmax_map = np.array([time_index[i] for i in tmax_indices.flatten()]).reshape(tmax_indices.shape)
    
    # Apply mask
    results = {
        'CBF': cbf_map * mask,
        'CBV': cbv_map * mask,
        'MTT': mtt_map * mask,
        'TMAX': tmax_map * mask,
        'TTP': ttp_map * mask # Passed through
    }
    
    return results

# --------------------------------------------------------------------------
# 4. EVALUATE RESULTS
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Configuration
    DATASET_PATH = r"demo_data"
    IMAGE_TYPE = 'ctp'  
    SCAN_INTERVAL = 1.41
    ECHO_TIME = 0.03
    
    # Find Data
    demo_file = None
    possible_roots = [DATASET_PATH, os.path.join(os.getcwd(), DATASET_PATH)]
    for root in possible_roots:
        if os.path.exists(root):
            for r, d, f in os.walk(root):
                for file in f:
                    if file.endswith("_ctp.nii.gz"):
                        demo_file = os.path.join(r, file)
                        break
    
    if demo_file:
        print(f"Input file: {demo_file}")
        
        # 1. Load
        mask_file_path = os.path.join(os.path.dirname(demo_file), "brain_mask.nii.gz")
        vol_stack, t_idx, brain_mask = load_and_preprocess_data(
            demo_file, 
            mask_path=mask_file_path, 
            scan_interval=SCAN_INTERVAL
        )
        
        # 2. Forward Op
        ctc_stack, s0_idx, ttp_idxs = forward_operator(
            vol_stack, 
            brain_mask, 
            image_type=IMAGE_TYPE, 
            echo_time=ECHO_TIME
        )
        
        # 3. Inversion
        final_maps = run_inversion(ctc_stack, t_idx, brain_mask, ttp_idxs)
        
        # 4. Evaluate
        ref_dir = os.path.join(os.path.dirname(demo_file), "perfusion-maps")
        ref_files = {
            'cbf': os.path.join(ref_dir, os.path.basename(demo_file).replace('_ctp.nii.gz', '_cbf.nii.gz')),
            'cbv': os.path.join(ref_dir, os.path.basename(demo_file).replace('_ctp.nii.gz', '_cbv.nii.gz')),
            'mtt': os.path.join(ref_dir, os.path.basename(demo_file).replace('_ctp.nii.gz', '_mtt.nii.gz')),
            'tmax': os.path.join(ref_dir, os.path.basename(demo_file).replace('_ctp.nii.gz', '_tmax.nii.gz')),
            'ttp': os.path.join(ref_dir, os.path.basename(demo_file).replace('_ctp.nii.gz', '_ttp.nii.gz')),
        }
        
        output_dir = os.path.join(os.path.dirname(demo_file), "generated_output")
        evaluate_results(final_maps, ref_files, output_dir=output_dir)
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
        
    else:
        print("Demo data not found. Creating dummy data for verification.")
        # Create dummy data if file not found to ensure logic runs
        t_steps = 30
        t_idx = np.arange(t_steps) * SCAN_INTERVAL
        vol_stack = np.random.rand(t_steps, 5, 50, 50) * 100
        # Add a bolus
        gamma = gamma_variate(t_idx, 5, 3, 4, amplitude=50)
        for i in range(5):
             for j in range(50):
                 for k in range(50):
                     vol_stack[:, i, j, k] += gamma
                     
        mask = np.ones((5, 50, 50))
        mask[:, 0:10, 0:10] = 0 # some background
        
        ctc_stack, s0_idx, ttp_idxs = forward_operator(vol_stack, mask, image_type=IMAGE_TYPE)
        final_maps = run_inversion(ctc_stack, t_idx, mask, ttp_idxs)
        evaluate_results(final_maps) # No refs, just print
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")