

# --- Extracted Dependencies ---

def downsample_image(img, factor):
    """Downsample image by a given factor using simple slicing."""
    if factor == 1:
        return img
    if len(img.shape) == 2:
        return img[::factor, ::factor]
    elif len(img.shape) == 3:
        return img[::factor, ::factor, :]
    else:
        return img
