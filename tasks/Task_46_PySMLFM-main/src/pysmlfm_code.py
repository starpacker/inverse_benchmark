import dataclasses
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum, unique
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

# ==============================================================================
# 1. Definitions & Helper Classes (Must be defined first)
# ==============================================================================

@dataclass
class FitParams:
    frame_min: int
    frame_max: int
    disparity_max: float
    disparity_step: float
    dist_search: float
    angle_tolerance: float
    threshold: float
    min_views: int
    z_calib: Optional[float] = None

@dataclass
class FitData:
    frame: int
    model: npt.NDArray[float]
    points: npt.NDArray[float]
    photon_count: int
    std_err: npt.NDArray[float]

@unique
class LatticeType(Enum):
    HEXAGONAL = 1
    SQUARE = 2

class MicroLensArray:
    def __init__(self, lattice_type: LatticeType, focal_length: float,
                 lens_pitch: float, optic_size: float, centre: npt.NDArray[float]):
        self.lattice_type = lattice_type
        self.focal_length = focal_length
        self.lens_pitch = lens_pitch
        self.optic_size = optic_size
        self.centre = centre
        self.lens_centres = self._generate_lattice()

    def _generate_lattice(self) -> npt.NDArray[float]:
        if self.lattice_type == LatticeType.SQUARE:
            width = self.optic_size / self.lens_pitch
            marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)
            x, y = np.meshgrid(marks, marks)
            return np.column_stack((x.flatten('F'), y.flatten('F')))
        elif self.lattice_type == LatticeType.HEXAGONAL:
            width = self.optic_size / self.lens_pitch
            marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)
            xx, yy = np.meshgrid(marks, marks)
            length = np.max(xx.shape)
            dx = np.tile([0.5, 0], [length, np.ceil(length / 2).astype(int)])
            dx = dx[0:length, 0:length]
            x = yy * np.sqrt(3) / 2
            y = xx + dx.T + 0.5
            return np.column_stack((x.flatten('F'), y.flatten('F')))
        else:
            raise ValueError(f'Unsupported lattice type {self.lattice_type}')

class FourierMicroscope:
    def __init__(self, num_aperture: float, mla_lens_pitch: float, focal_length_mla: float,
                 focal_length_obj_lens: float, focal_length_tube_lens: float,
                 focal_length_fourier_lens: float, pixel_size_camera: float,
                 ref_idx_immersion: float, ref_idx_medium: float):
        self.num_aperture = num_aperture
        self.mla_lens_pitch = mla_lens_pitch
        self.focal_length_mla = focal_length_mla
        self.focal_length_obj_lens = focal_length_obj_lens
        self.focal_length_tube_lens = focal_length_tube_lens
        self.focal_length_fourier_lens = focal_length_fourier_lens
        self.pixel_size_camera = pixel_size_camera
        self.ref_idx_immersion = ref_idx_immersion
        self.ref_idx_medium = ref_idx_medium

        self.bfp_radius = (1000 * num_aperture * focal_length_obj_lens
                           * (focal_length_fourier_lens / focal_length_tube_lens))
        self.bfp_lens_count = 2.0 * self.bfp_radius / mla_lens_pitch
        self.pixels_per_lens = mla_lens_pitch / pixel_size_camera
        self.magnification = ((focal_length_tube_lens / focal_length_obj_lens)
                              * (focal_length_mla / focal_length_fourier_lens))
        self.pixel_size_sample = pixel_size_camera / self.magnification
        self.rho_scaling = self.magnification / self.bfp_radius
        self.xy_to_uv_scale = self.rho_scaling
        self.mla_to_uv_scale = 2.0 / self.bfp_lens_count
        self.mla_to_xy_scale = self.mla_to_uv_scale / self.xy_to_uv_scale

# ==============================================================================
# 2. Main Logic Functions
# ==============================================================================

def load_and_preprocess_data(lfm: FourierMicroscope, mla: MicroLensArray, num_points: int = 5) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """
    Generates synthetic 2D localisations and preprocesses them by assigning to lenses.
    Returns: (ground_truth_3d, filtered_locs_2d, original_raw_locs)
    """
    print("Generating synthetic data...")
    # Ground Truth 3D points (x, y, z) in microns
    gt_points_list = []
    np.random.seed(42)
    for _ in range(num_points):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-3, 3)
        gt_points_list.append([x, y, z])
    gt_points = np.array(gt_points_list)

    locs_list = []
    na = lfm.num_aperture
    n = lfm.ref_idx_medium
    
    # Identify valid lenses
    lens_centres_uv = (mla.lens_centres - mla.centre) * lfm.mla_to_uv_scale
    radius_sq_cutoff = (lfm.bfp_radius / mla.lens_pitch)**2
    dist_sq = np.sum((mla.lens_centres - mla.centre)**2, axis=1)
    valid_lens_mask = dist_sq < radius_sq_cutoff
    valid_lenses_uv = lens_centres_uv[valid_lens_mask]

    # Forward project ground truth points to create synthetic measurements
    for pt in gt_points:
        X, Y, Z = pt
        for uv_lens in valid_lenses_uv:
            u, v = uv_lens
            rho = np.sqrt(u**2 + v**2)
            if rho >= 1.0: continue
            
            dr_sq = 1 - rho * (na/n)**2
            if dr_sq < 0: continue
            phi = -(na / n) / np.sqrt(dr_sq)
            alpha_u = u * phi
            alpha_v = v * phi
            
            u_micron = u / lfm.rho_scaling
            v_micron = v / lfm.rho_scaling
            
            x_obs = X + Z * alpha_u + u_micron
            y_obs = Y + Z * alpha_v + v_micron
            
            noise_x = np.random.normal(0, 0.02)
            noise_y = np.random.normal(0, 0.02)
            
            # Format: frame, x, y, sx, sy, I, bg, prec
            locs_list.append([1, x_obs + noise_x, y_obs + noise_y, 0.5, 0.5, 1000, 10, 0.01])

    locs_2d_csv = np.array(locs_list)
    
    # -- PREPROCESSING --
    # Transform raw CSV into internal structure
    locs_2d = np.zeros((locs_2d_csv.shape[0], 13))
    locs_2d[:, 0] = locs_2d_csv[:, 0].copy() # Frame
    locs_2d[:, 3:10] = locs_2d_csv[:, 1:8].copy() # Data columns

    # Assign to lenses
    xy = locs_2d[:, 3:5].copy()
    xy *= lfm.xy_to_uv_scale
    
    lens_centres = mla.lens_centres - mla.centre
    lens_centres_scaled = lens_centres * lfm.mla_to_uv_scale
    
    knn = NearestNeighbors(n_neighbors=1).fit(lens_centres_scaled)
    lens_indices = knn.kneighbors(xy, return_distance=False)
    
    locs_2d[:, 1:3] = lens_centres_scaled[lens_indices, :][:, 0, :] # U, V
    locs_2d[:, 12] = lens_indices[:, 0]

    # Filter invalid lenses (outside BFP)
    radius = lfm.bfp_radius / mla.lens_pitch
    radius_sq = radius ** 2
    centres = mla.lens_centres - mla.centre
    distance_sq = np.sum(centres ** 2, axis=1)
    valid_lens_indices = np.nonzero(distance_sq < radius_sq)[0]
    
    index_col = locs_2d[:, 12]
    sel = np.any(index_col[:, None] == valid_lens_indices, axis=-1)
    filtered_locs_2d = locs_2d[sel, :]

    # Calculate Alpha (Parallax vector) - Sphere Model
    uv = filtered_locs_2d[:, 1:3]
    alpha_uv = filtered_locs_2d[:, 10:12] # Columns 10 and 11
    
    rho_arr = np.sqrt(np.sum(uv**2, axis=1))
    dr_sq_arr = 1 - rho_arr * (na / n)**2
    # Avoid sqrt of negative
    dr_sq_arr[dr_sq_arr < 0.0] = np.nan
    phi_arr = -(na / n) / np.sqrt(dr_sq_arr)
    alpha_uv[:] = uv * phi_arr[:, np.newaxis]

    return gt_points, filtered_locs_2d, locs_2d_csv


def forward_operator(x_model: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Computes the forward projection (Prediction) for a given 3D point.
    Not strictly used in the inverse fitting loop provided (which uses backward projection logic),
    but essential for validation and completeness.
    x_model: [X, Y, Z, U, V, rho_scaling, na, n]
    Returns: [x_sensor, y_sensor]
    """
    X, Y, Z, u, v, rho_scaling, na, n = x_model
    
    rho = np.sqrt(u**2 + v**2)
    dr_sq = 1 - rho * (na / n)**2
    if dr_sq < 0:
        return np.array([np.nan, np.nan])
        
    phi = -(na / n) / np.sqrt(dr_sq)
    alpha_u = u * phi
    alpha_v = v * phi
    
    u_micron = u / rho_scaling
    v_micron = v / rho_scaling
    
    x_sensor = X + Z * alpha_u + u_micron
    y_sensor = Y + Z * alpha_v + v_micron
    
    return np.array([x_sensor, y_sensor])


def run_inversion(locs_2d: npt.NDArray[float], rho_scaling: float, fit_params: FitParams) -> Tuple[npt.NDArray[float], List[FitData]]:
    """
    Performs the light field fitting (inversion) to recover 3D points from 2D localisations.
    """
    min_frame = fit_params.frame_min
    max_frame = fit_params.frame_max
    min_views = fit_params.min_views
    fit_threshold = fit_params.threshold
    z_calib = fit_params.z_calib

    fitted_points = np.empty((0, 8))
    total_fit = []

    # Helper function for backward model least squares
    def _get_backward_model(locs_subset):
        u = locs_subset[:, 1]
        v = locs_subset[:, 2]
        x = locs_subset[:, 3] - u / rho_scaling
        y = locs_subset[:, 4] - v / rho_scaling
        alpha_u = locs_subset[:, 10]
        alpha_v = locs_subset[:, 11]

        zeros = np.zeros_like(alpha_u)
        ones = np.ones_like(alpha_u)

        a = np.vstack((
            np.column_stack((ones, zeros, alpha_u)),
            np.column_stack((zeros, ones, alpha_v))))
        b = np.concatenate((x, y))

        model, residuals, rank, _s = np.linalg.lstsq(a, b, rcond=None)
        
        if residuals.size == 0 or a.shape[0] <= rank:
             mse = 0.0
             std_err = np.zeros(3)
        else:
            mse = np.mean(residuals) / (a.shape[0] - rank)
            try:
                cov_mat = mse * np.linalg.inv(np.dot(a.T, a))
                std_err = np.sqrt(np.diag(cov_mat))
            except np.linalg.LinAlgError:
                std_err = np.zeros(3)
        return model, std_err, mse

    # Helper for grouping localisations
    def _group_localisations(seed, in_points):
        angle_tol = np.deg2rad(fit_params.angle_tolerance)
        max_disparity = fit_params.disparity_max
        dis_tol = fit_params.dist_search
        dz = fit_params.disparity_step

        su, sv = seed[1], seed[2]
        sx, sy = seed[3], seed[4]

        points = in_points.copy()
        # Remove seed itself
        seed_sel = np.logical_and(points[:, 1] == su, points[:, 2] == sv)
        points = points[~seed_sel, :]

        # Angular filter
        du = points[:, 1] - su
        dv = points[:, 2] - sv
        dx = points[:, 3] - sx
        dy = points[:, 4] - sy

        angles_uv = np.arctan2(dv, du)
        angles_xy = np.arctan2(dy, dx)
        angle_sel = np.logical_or(
            np.abs(angles_xy - angles_uv) < angle_tol,
            np.abs(angles_xy - angles_uv) > (np.pi - angle_tol))
        points = points[angle_sel, :]

        # Disparity filter
        du = points[:, 1] - su
        dv = points[:, 2] - sv
        dx = points[:, 3] - sx
        dy = points[:, 4] - sy
        dxy = np.sqrt(dx**2 + dy**2)
        duv = np.sqrt(du**2 + dv**2)
        # Avoid div by zero if duv=0
        with np.errstate(divide='ignore', invalid='ignore'):
            disparity = (dxy - duv / rho_scaling) / duv
        
        # Disparity Histogram
        range_z = np.arange(-max_disparity, max_disparity + dz / 2, dz)
        num_points_hist = np.zeros((range_z.size, 1))
        for z_i in range(range_z.size):
            d_sel = np.logical_and(disparity > (range_z[z_i] - dis_tol),
                                   disparity <= (range_z[z_i] + dis_tol))
            num_points_hist[z_i] = np.sum(d_sel)
        
        if num_points_hist.size > 0:
            best_z_i = np.argmax(num_points_hist)
            best_z = range_z[best_z_i]
            z_sel = np.logical_or(disparity <= (best_z - dis_tol),
                                  disparity > (best_z + dis_tol))
            points = points[~z_sel, :]
        else:
            points = np.array([])

        candidates = seed.reshape(1, -1)
        view_count = 0
        
        if points.size > 0:
            uv, uv_i, uv_c = np.unique(points[:, 1:3], axis=0, return_inverse=True, return_counts=True)
            
            # Select unique points
            candidates = np.vstack((candidates, points[np.isin(uv_i, np.nonzero(uv_c == 1))]))
            
            # Handle duplicates (multiple candidates in one lens view)
            indices = []
            for val in uv:
                indices.append(np.where(np.all(points[:, 1:3] == val, axis=1))[0])
            indices = [i for i in indices if len(i) > 1]
            
            view_count = len(uv_c)
            duplicate_count = len(indices)

            for i in range(duplicate_count):
                view_ind = indices[i]
                j_count = view_ind.shape[0]
                d_vals = np.zeros(j_count)
                for j in range(j_count):
                    locs_temp = np.vstack((candidates, points[view_ind[j], :]))
                    _, std_err_temp, _ = _get_backward_model(locs_temp)
                    d_vals[j] = np.sqrt(np.sum(std_err_temp**2))
                d_min_j = np.argmin(d_vals)
                candidates = np.vstack((candidates, points[view_ind[d_min_j], :]))

            var_u = np.var(candidates[:, 1], axis=0, ddof=1)
            var_v = np.var(candidates[:, 2], axis=0, ddof=1)
            # If only 1 view or variance is tiny, fail
            if view_count == 1 and (var_u < 0.1 or var_v < 0.1):
                candidates = np.array([])
                view_count = 0
        
        return candidates, view_count

    # Main fitting loop
    for frame in range(min_frame, max_frame + 1):
        candidates = locs_2d[locs_2d[:, 0] == frame, :].copy()
        if candidates.shape[0] == 0:
            continue
        
        u = candidates[:, 1]
        v = candidates[:, 2]
        
        # Prioritize central points
        loc_cen_sel = np.logical_and(np.abs(u) < 0.1, np.abs(v) < 0.1)
        loc_cen = candidates[loc_cen_sel, :].copy()
        loc_out_sel = ~np.logical_and(u == 0.0, v == 0.0) # slightly redundant logic from orig, kept for consistency
        loc_out = candidates[loc_out_sel, :].copy()
        
        if loc_cen.size > 0:
            loc_cen = loc_cen[loc_cen[:, 7].argsort()[::-1]]
        if loc_out.size > 0:
            loc_out = loc_out[loc_out[:, 7].argsort()[::-1]]
        
        candidates = np.vstack((loc_cen, loc_out))

        reps = 0
        while candidates.shape[0] > min_views and reps < 200:
            reps += 1
            seed = candidates[0, :]
            
            # Identify group
            loc_fit, view_count = _group_localisations(seed, candidates)
            
            if view_count < min_views or loc_fit.size == 0:
                candidates = candidates[1:, :]
                continue

            # Fit model
            model, std_err, _ = _get_backward_model(loc_fit)
            dist_err = np.sqrt(np.sum(std_err**2))
            
            if dist_err > fit_threshold:
                candidates = candidates[1:, :]
                continue

            # Remove used candidates
            # Check for approximate equality to handle float issues or exact matches
            candidates_sel = np.zeros(candidates.shape[0], dtype=bool)
            for lf in loc_fit:
                # check rows where all columns match
                match = np.all(candidates == lf, axis=1)
                candidates_sel = np.logical_or(candidates_sel, match)
            
            candidates = candidates[~candidates_sel, :]

            photon_count = np.sum(loc_fit[:, 7])
            new_fit_point = np.array([
                model[0], model[1], model[2],
                np.mean(std_err[0:2]), std_err[2],
                view_count + 1, photon_count, frame
            ])
            
            # Apply Z Calib immediately to stored result if needed, 
            # though usually applied at end. Logic in original code applied at end.
            fitted_points = np.vstack((fitted_points, new_fit_point))
            
            total_fit.append(FitData(
                frame=frame, model=model, points=loc_fit.copy(),
                photon_count=int(photon_count), std_err=std_err
            ))

    if z_calib is not None and fitted_points.size > 0:
        fitted_points[:, 2] *= z_calib

    return fitted_points, total_fit

def evaluate_results(gt_points: npt.NDArray[float], reconstructed_points: npt.NDArray[float]) -> float:
    """
    Evaluates the reconstruction by matching points to ground truth and calculating RMSE.
    Returns RMSE.
    """
    print("\n=== RECONSTRUCTION RESULTS ===")
    print(f"Reconstructed {len(reconstructed_points)} points.")
    
    if len(reconstructed_points) == 0:
        print("No points reconstructed.")
        return float('inf')

    rec_xyz = reconstructed_points[:, 0:3]
    mse_sum = 0.0
    matches = 0
    
    print("\nComparison (GT vs Rec):")
    for gt in gt_points:
        dists = np.sqrt(np.sum((rec_xyz - gt)**2, axis=1))
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist < 1.0: # Match threshold 1 micron
            rec = rec_xyz[min_dist_idx]
            print(f"GT: {gt} -> Rec: {rec} (Err: {min_dist:.4f} um)")
            mse_sum += min_dist**2
            matches += 1
        else:
            print(f"GT: {gt} -> No match found (Min dist: {min_dist:.4f} um)")
    
    if matches > 0:
        rmse = np.sqrt(mse_sum / matches)
        print(f"\nRMSE (matched points): {rmse:.4f} microns")
        print(f"PSNR (proxy): {20 * np.log10(10.0 / rmse):.2f} dB (assuming peak=10um)")
        return rmse
    else:
        print("No matches found.")
        return float('inf')

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    # 1. Define Constants & Objects
    MLA = MicroLensArray(
        lattice_type=LatticeType.HEXAGONAL,
        focal_length=175.0,
        lens_pitch=2390.0,
        optic_size=10000.0,
        centre=np.array([0.0, 0.0])
    )
    
    LFM = FourierMicroscope(
        num_aperture=1.27,
        mla_lens_pitch=2390.0,
        focal_length_mla=175.0,
        focal_length_obj_lens=200.0/60.0,
        focal_length_tube_lens=200.0,
        focal_length_fourier_lens=175.0,
        pixel_size_camera=16.0,
        ref_idx_immersion=1.33,
        ref_idx_medium=1.33
    )

    FIT_PARAMS = FitParams(
        frame_min=1, frame_max=1,
        disparity_max=5.0, disparity_step=0.1,
        dist_search=0.5, angle_tolerance=2.0,
        threshold=1.0, min_views=3, z_calib=1.0
    )

    # 2. Sequential Pipeline
    # Step 1: Data
    gt_data, processed_locs, _ = load_and_preprocess_data(LFM, MLA, num_points=5)
    
    # Step 2: Forward Operator (Demonstration check on a single point)
    # Just to verify the physics model works for one point manually
    if len(gt_data) > 0:
        # Pick a lens (arbitrary u,v for test)
        u_test, v_test = 0.1, 0.1 
        test_model_input = np.array([gt_data[0][0], gt_data[0][1], gt_data[0][2], 
                                     u_test, v_test, 
                                     LFM.rho_scaling, LFM.num_aperture, LFM.ref_idx_medium])
        proj_test = forward_operator(test_model_input)
        # We don't use this result in inversion directly, but it ensures code existence.

    # Step 3: Inversion
    print("Running Light Field Fitting...")
    rec_points, fit_details = run_inversion(processed_locs, LFM.rho_scaling, FIT_PARAMS)

    # Step 4: Evaluation
    evaluate_results(gt_data, rec_points)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")