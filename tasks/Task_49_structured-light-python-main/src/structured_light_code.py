import sys
import os
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
def load_and_preprocess_data(
    width: int,
    height: int,
    frequencies: List[float],
    shift_num: int
) -> Dict[str, Any]:
    """
    Generates simulation configuration and phase-shifting patterns.
    Acts as the data loader for this simulation-based problem.
    """
    
    # Helper to create patterns
    def create_psp_patterns(w, h, freqs, shifts, vertical=True):
        patterns = []
        # Standard N-step phase shifting angles
        phase_shifts_angles = [2 * np.pi / shifts * i for i in range(shifts)]
        
        if vertical:
            length = w
        else:
            length = h
            
        x = np.linspace(0, length, length)
        
        freq_patterns = []
        for frequency in freqs:
            step_patterns = []
            for phase_shift in phase_shifts_angles:
                # Cosine pattern generation: 0.5 + 0.5 * cos(...)
                cos_val = 0.5 + 0.5 * np.cos(2 * np.pi * frequency * (x / length) - phase_shift)
                if vertical:
                    pat = np.tile(cos_val, (h, 1))
                else:
                    pat = np.tile(cos_val.reshape((-1, 1)), (1, w))
                step_patterns.append(pat)
            freq_patterns.append(step_patterns)
        return freq_patterns, phase_shifts_angles

    # Generate patterns
    patterns_v, _ = create_psp_patterns(width, height, frequencies, shift_num, vertical=True)
    patterns_h, phase_shifts = create_psp_patterns(width, height, frequencies, shift_num, vertical=False)

    # Define calibration data (Hardcoded for simulation context)
    f = 1000.0
    cx = width / 2.0
    cy = height / 2.0
    mtx = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist = np.zeros(5)
    R = np.eye(3)
    # Baseline T = [100, 0, 0], Camera 2 is shifted by 100 units relative to Camera 1
    T = np.array([[100.0], [0.0], [0.0]]) 

    calibration_data = {
        'camera_0': {'mtx': mtx, 'dist': dist},
        'camera_1': {'mtx': mtx, 'dist': dist},
        'R': R,
        'T': T
    }

    data = {
        'frequencies': frequencies,
        'phase_shifts': phase_shifts,
        'patterns_vertical': patterns_v,
        'patterns_horizontal': patterns_h,
        'calibration': calibration_data,
        'image_shape': (height, width)
    }
    
    return data

# ==========================================
# 2. FORWARD OPERATOR
# ==========================================
def forward_operator(
    patterns_dict: Dict[str, List[List[np.ndarray]]], 
    disparity_pixel_shift: int = 20
) -> Dict[str, List[List[List[np.ndarray]]]]:
    """
    Simulates the projection and capture process.
    Input: Dictionary of projection patterns (vertical/horizontal).
    Output: Captured images for Camera 0 and Camera 1.
    Physics: Cam 0 sees pattern exactly. Cam 1 sees it shifted (disparity).
    """
    
    captured_data = {
        'cam1_vertical': [], 'cam2_vertical': [],
        'cam1_horizontal': [], 'cam2_horizontal': []
    }
    
    # Process Vertical Fringes
    # Structure: patterns_dict['vertical'][freq_idx][shift_idx]
    for freq_idx, pattern_group in enumerate(patterns_dict['patterns_vertical']):
        c1_group = []
        c2_group = []
        for pattern in pattern_group:
            # Camera 0: Identity (Projector view in this simplified model)
            img1 = pattern.copy()
            
            # Camera 1: Simulated Disparity Shift
            img2 = np.roll(img1, disparity_pixel_shift, axis=1)
            # Handle edge artifacts from roll
            if disparity_pixel_shift > 0:
                img2[:, :disparity_pixel_shift] = 0
            elif disparity_pixel_shift < 0:
                img2[:, disparity_pixel_shift:] = 0
                
            c1_group.append(img1)
            c2_group.append(img2)
        captured_data['cam1_vertical'].append(c1_group)
        captured_data['cam2_vertical'].append(c2_group)

    # Process Horizontal Fringes
    for freq_idx, pattern_group in enumerate(patterns_dict['patterns_horizontal']):
        c1_group = []
        c2_group = []
        for pattern in pattern_group:
            img1 = pattern.copy()
            
            # Camera 1: Same horizontal shift (epipolar geometry assumption)
            img2 = np.roll(img1, disparity_pixel_shift, axis=1)
            if disparity_pixel_shift > 0:
                img2[:, :disparity_pixel_shift] = 0
            elif disparity_pixel_shift < 0:
                img2[:, disparity_pixel_shift:] = 0
                
            c1_group.append(img1)
            c2_group.append(img2)
        captured_data['cam1_horizontal'].append(c1_group)
        captured_data['cam2_horizontal'].append(c2_group)
        
    return captured_data

# ==========================================
# 3. RUN INVERSION
# ==========================================
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

# ==========================================
# 4. EVALUATE RESULTS
# ==========================================
def evaluate_results(points_3d: np.ndarray, expected_depth: float = 5000.0) -> None:
    """
    Calculates statistics on reconstructed point cloud and compares to theoretical model.
    """
    if points_3d.shape[0] == 0:
        print("EVALUATION FAILED: No 3D points reconstructed.")
        return

    z_vals = points_3d[:, 2]
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    
    error = abs(z_mean - expected_depth)
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Reconstructed Points: {points_3d.shape[0]}")
    print(f"Mean Z Depth:         {z_mean:.4f} mm")
    print(f"Std Dev Z:            {z_std:.4f} mm")
    print(f"Theoretical Z:        {expected_depth:.4f} mm")
    print(f"Absolute Error:       {error:.4f} mm")
    
    # Tolerance check
    if error < 50.0: # Generous tolerance for discrete simulation
        print("STATUS: SUCCESS (Within tolerance)")
    else:
        print("STATUS: WARNING (High deviation)")
        
    # Calculate simple PSNR proxy (Linearity check logic from original code)
    # Since we don't have the phase map here, we check planarity of Z
    # Ideal surface is flat (Z = constant)
    mse = np.mean((z_vals - z_mean)**2)
    if mse == 0:
        psnr = 100
    else:
        max_val = np.max(z_vals)
        psnr = 20 * math.log10(max_val / math.sqrt(mse))
    
    print(f"Surface Planarity PSNR: {psnr:.2f} dB")
    print("="*30 + "\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # Configuration
    PROJ_W, PROJ_H = 1280, 720
    FREQUENCIES = [1, 4, 12]
    SHIFTS_NUM = 4
    PIXEL_DISPARITY = 20
    
    # 1. Load Data
    print("[1/4] Loading and Preprocessing Data...")
    sim_data = load_and_preprocess_data(PROJ_W, PROJ_H, FREQUENCIES, SHIFTS_NUM)
    
    # 2. Forward Model (Simulation)
    print("[2/4] Running Forward Operator (Simulation)...")
    captured_images = forward_operator(
        sim_data, 
        disparity_pixel_shift=PIXEL_DISPARITY
    )
    
    # 3. Inverse Problem
    print("[3/4] Running Inversion (Phase Unwrapping & Triangulation)...")
    reconstructed_points = run_inversion(
        captured_images,
        sim_data['frequencies'],
        sim_data['phase_shifts'],
        sim_data['calibration']
    )
    
    # 4. Evaluation
    print("[4/4] Evaluating Results...")
    # Theoretical depth Z = f * Baseline / Disparity
    # f=1000, Baseline=100, Disparity=20
    THEORETICAL_Z = (1000.0 * 100.0) / float(PIXEL_DISPARITY)
    
    evaluate_results(reconstructed_points, expected_depth=THEORETICAL_Z)

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")