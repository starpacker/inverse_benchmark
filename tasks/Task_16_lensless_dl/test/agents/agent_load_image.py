import numpy as np


# --- Extracted Dependencies ---

def load_image(fp, dtype="float32"):
    """Load an image file and return as numpy array."""
    from PIL import Image
    img = Image.open(fp)
    img_array = np.array(img).astype(dtype)
    return img_array
