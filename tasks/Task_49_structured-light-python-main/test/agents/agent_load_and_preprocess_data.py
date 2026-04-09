import numpy as np

from typing import List, Tuple, Optional, Any, Dict

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
