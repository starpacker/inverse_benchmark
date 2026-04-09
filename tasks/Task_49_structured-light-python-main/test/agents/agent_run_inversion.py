import numpy as np

import cv2

from typing import List, Tuple, Optional, Any, Dict

def run_inversion(
    captured_data: Dict[str, Any],
    frequencies: List[float],
    phase_shifts: List[float],
    calibration: Dict[str, Any]
) -> np.ndarray:
    """
    Performs Phase Wrapping, Unwrapping, Phasogrammetry (Matching), and Triangulation.
    """

    # --- 3a. Phase Calculation Helper ---
    def calculate_phase(images_freq_group, shifts):
        # images_freq_group: List[np.ndarray] (N steps)
        imgs = np.array(images_freq_group)
        shifts_arr = np.array(shifts).reshape((-1, 1, 1))
        
        # Least squares phase extraction (or standard N-step formula)
        # formula: tan(phi) = - sum(I * sin(delta)) / sum(I * cos(delta)) 
        # Note: The input code used a sum based approach equivalent to DFT 
        
        sin_part = np.sum(imgs * np.sin(shifts_arr), axis=0)
        cos_part = np.sum(imgs * np.cos(shifts_arr), axis=0)
        
        wrapped_phase = np.arctan2(sin_part, cos_part)
        magnitude = 2 * np.sqrt(sin_part**2 + cos_part**2) / len(images_freq_group)
        
        return wrapped_phase, magnitude

    # --- 3b. Temporal Phase Unwrapping Helper ---
    def unwrap_phases(wrapped_phases_list, freqs):
        # Multi-frequency temporal unwrapping
        # Start with lowest frequency (usually freq=1, where phase range is -pi to pi over whole image)
        
        unwrapped = wrapped_phases_list[0] # Base phase
        
        for i in range(1, len(freqs)):
            p_low = unwrapped
            p_high = wrapped_phases_list[i]
            lambda_l = 1.0 / freqs[i-1]
            lambda_h = 1.0 / freqs[i]
            
            # k = round( ( (lambda_l/lambda_h) * p_low - p_high ) / 2pi )
            ratio = lambda_l / lambda_h
            k = np.round((ratio * p_low - p_high) / (2 * np.pi))
            unwrapped = p_high + 2 * np.pi * k
            
        return unwrapped

    # --- 3c. Process All Cameras/Orientations ---
    # We need unwrapped phase maps for: 
    # Cam1 Vertical (p1_v), Cam2 Vertical (p2_v) -> X coordinate encoding
    # Cam1 Horizontal (p1_h), Cam2 Horizontal (p2_h) -> Y coordinate encoding
    
    phase_maps = {}
    keys_map = [
        ('cam1_vertical', 'p1_v'), ('cam2_vertical', 'p2_v'),
        ('cam1_horizontal', 'p1_h'), ('cam2_horizontal', 'p2_h')
    ]
    
    for input_key, output_key in keys_map:
        wrapped_list = []
        for f_idx, imgs in enumerate(captured_data[input_key]):
            w_phase, _ = calculate_phase(imgs, phase_shifts)
            wrapped_list.append(w_phase)
        
        unwrapped = unwrap_phases(wrapped_list, frequencies)
        phase_maps[output_key] = unwrapped

    # --- 3d. Phasogrammetry (Matching) ---
    p1_h = phase_maps['p1_h'] # Y-encoding Cam 1
    p1_v = phase_maps['p1_v'] # X-encoding Cam 1
    p2_h = phase_maps['p2_h'] # Y-encoding Cam 2
    p2_v = phase_maps['p2_v'] # X-encoding Cam 2
    
    h, w = p1_h.shape
    step = 50 # Subsampling for speed
    
    # Define ROI (crop borders)
    roi_margin = 100
    xx = np.arange(roi_margin, w - roi_margin, step)
    yy = np.arange(roi_margin, h - roi_margin, step)
    
    pts1_list = []
    pts2_list = []

    # Optimization: vectorize finding corresponding points?
    # The legacy code iterates. We must implement the logic.
    # To find match for (x, y) in Cam1:
    # 1. Get phase values phi_x, phi_y at (x,y) in Cam1.
    # 2. Find (u, v) in Cam2 such that P2_v(u,v) approx phi_x AND P2_h(u,v) approx phi_y.
    
    # Pre-computation for matching speedup isn't strictly forbidden but let's stick to the logic provided.
    # However, iterating is slow in Python. Let's do a constrained search based on epipolar geometry 
    # implicitly provided by the simulation (horizontal shift only).
    
    for y in yy:
        for x in xx:
            target_ph_h = p1_h[y, x]
            target_ph_v = p1_v[y, x]
            
            # In rectified stereo (or this simulation), corresponding point is on the same scanline y
            # So we only search along row y in Cam2 (or close to it)
            # Scan a window around y
            y_search_start = max(0, y - 2)
            y_search_end = min(h, y + 3)
            
            # Extract strips
            strip_p2_h = p2_h[y_search_start:y_search_end, :]
            strip_p2_v = p2_v[y_search_start:y_search_end, :]
            
            # Find candidate columns where phase matches
            # Metric: Euclidean distance in phase space
            dist_map = np.sqrt((strip_p2_h - target_ph_h)**2 + (strip_p2_v - target_ph_v)**2)
            
            min_val = np.min(dist_map)
            if min_val < 0.2: # Phase matching threshold
                # Get local coordinates of minimum
                loc_y, loc_x = np.unravel_index(np.argmin(dist_map), dist_map.shape)
                match_x = loc_x
                match_y = y_search_start + loc_y
                
                pts1_list.append([x, y])
                pts2_list.append([match_x, match_y])
                
    pts1 = np.array(pts1_list, dtype=np.float32)
    pts2 = np.array(pts2_list, dtype=np.float32)

    if len(pts1) == 0:
        return np.zeros((0, 3))

    # --- 3e. Triangulation ---
    cam1_mtx = calibration['camera_0']['mtx']
    cam2_mtx = calibration['camera_1']['mtx']
    dist1 = calibration['camera_0']['dist']
    dist2 = calibration['camera_1']['dist']
    R_mat = calibration['R']
    T_vec = calibration['T']
    
    # Projection matrices
    # P1 = K1 * [I | 0]
    P1 = np.dot(cam1_mtx, np.hstack((np.eye(3), np.zeros((3, 1)))))
    # P2 = K2 * [R | T]
    P2 = np.dot(cam2_mtx, np.hstack((R_mat, T_vec)))
    
    # Undistort points
    # Reshape to (N, 1, 2) for cv2
    pts1_cv = np.ascontiguousarray(pts1[:, :2]).reshape((pts1.shape[0], 1, 2))
    pts2_cv = np.ascontiguousarray(pts2[:, :2]).reshape((pts2.shape[0], 1, 2))
    
    pts1_undist = cv2.undistortPoints(pts1_cv, cam1_mtx, dist1, P=cam1_mtx)
    pts2_undist = cv2.undistortPoints(pts2_cv, cam2_mtx, dist2, P=cam2_mtx)
    
    # Triangulate
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)
    
    # Convert from homogeneous
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)
    points_3d = points_3d.reshape(-1, 3)
    
    return points_3d
