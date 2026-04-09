import numpy as np

import cv2

import os

def load_and_preprocess_data(image_path, target_shape=None):
    """
    Loads the ground truth amplitude, synthesizes phase, and creates the complex field.
    
    Returns:
        field_input (complex np.array): The complex object field.
        gt_amp (np.array): Ground truth amplitude [0, 1].
        gt_phase (np.array): Ground truth phase.
    """
    print(f"Loading Ground Truth Image: {image_path}")
    
    if not os.path.exists(image_path):
        # Trying a few fallbacks as per original logic if exact path fails, 
        # though the caller should ideally provide a valid path.
        # Here we just raise error if it really doesn't exist to be strict.
        raise FileNotFoundError(f"Error: File not found at {image_path}")

    gt_amp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gt_amp is None:
        raise ValueError("Error: Failed to read image.")
        
    # Resize if target_shape is provided (optional for flexibility)
    if target_shape is not None:
        gt_amp = cv2.resize(gt_amp, (target_shape[1], target_shape[0]))

    # Normalize to [0, 1]
    gt_amp = gt_amp.astype(np.float32) / 255.0
    
    # Generate Synthetic Phase (e.g., spherical phase)
    M, N = gt_amp.shape
    x = np.arange(0, N, 1)
    y = np.arange(0, M, 1)
    X, Y = np.meshgrid(x - N/2, y - M/2)
    # A simple phase lens-like structure
    gt_phase = 5 * np.exp(-(X**2 + Y**2) / (2 * (200**2))) 
    
    # Create Complex Object
    field_input = gt_amp * np.exp(1j * gt_phase)
    
    return field_input, gt_amp, gt_phase
