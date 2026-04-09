import numpy as np


# --- Extracted Dependencies ---

def pad_array(v, setup):
    """Pad array to padded_shape."""
    padded_shape = setup["padded_shape"]
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    
    if len(v.shape) == 5:
        batch_size = v.shape[0]
        shape = [batch_size] + padded_shape
    elif len(v.shape) == 4:
        shape = padded_shape
    else:
        raise ValueError("Expected 4D or 5D tensor")
    
    vpad = np.zeros(shape).astype(v.dtype)
    vpad[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :] = v
    return vpad
