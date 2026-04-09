from dataclasses import asdict, dataclass, field, fields, is_dataclass

from typing import Callable, List, Optional, Tuple

import numpy as np

import numpy.typing as npt

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

class FitData:
    frame: int
    model: npt.NDArray[float]
    points: npt.NDArray[float]
    photon_count: int
    std_err: npt.NDArray[float]

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
