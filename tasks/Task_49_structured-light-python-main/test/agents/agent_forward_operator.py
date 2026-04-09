import numpy as np

from typing import List, Tuple, Optional, Any, Dict

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
