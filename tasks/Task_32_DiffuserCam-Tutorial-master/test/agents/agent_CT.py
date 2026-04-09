import numpy as np

def CT(b, full_size, sensor_size):
    # Transpose of Crop (Zero Pad)
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    # Create full zero array and place b in center
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top+sensor_size[0], pad_left:pad_left+sensor_size[1]] = b
    return out
