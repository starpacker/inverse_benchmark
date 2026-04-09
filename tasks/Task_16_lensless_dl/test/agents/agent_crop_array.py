

# --- Extracted Dependencies ---

def crop_array(x, setup):
    """Crop array from padded_shape to original shape."""
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    return x[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :]
