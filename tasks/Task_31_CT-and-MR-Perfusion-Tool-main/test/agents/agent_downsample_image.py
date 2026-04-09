import numpy as np

import warnings

warnings.filterwarnings("ignore")

def downsample_image(image_array, time_index=None, t_limit=100, z_limit=25, y_limit=250, x_limit=250):
    def generate_indices(original_size, target_size):
        if target_size >= original_size:
            return np.arange(original_size)
        return np.linspace(0, original_size - 1, target_size, dtype=int)
    
    def calculate_spatial_dims(x_size, y_size, x_limit, y_limit):
        if x_size <= x_limit and y_size <= y_limit:
            return x_size, y_size
        x_scale = x_limit / x_size if x_size > x_limit else 1.0
        y_scale = y_limit / y_size if y_size > y_limit else 1.0
        scale_factor = min(x_scale, y_scale)
        return int(x_size * scale_factor), int(y_size * scale_factor)
    
    if image_array.ndim == 4:
        t_size, z_size, y_size, x_size = image_array.shape
        new_t = min(t_size, t_limit)
        new_z = min(z_size, z_limit)
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit)

        indices = [
            generate_indices(t_size, new_t),
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]
        downsampled = image_array[np.ix_(*indices)]
        if time_index is not None:
            new_time = [time_index[i] for i in indices[0]]
            return downsampled, new_time
        return downsampled

    elif image_array.ndim == 3:
        z_size, y_size, x_size = image_array.shape
        new_z = min(z_size, z_limit)
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit)
        indices = [
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]
        return image_array[np.ix_(*indices)]
    
    return image_array
