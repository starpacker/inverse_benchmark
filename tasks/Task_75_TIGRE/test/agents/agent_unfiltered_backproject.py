import matplotlib

matplotlib.use('Agg')

import os

from skimage.transform import iradon

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def unfiltered_backproject(sinogram, angles_deg, img_size):
    """
    Unfiltered backprojection using iradon with filter_name=None.
    Used as the adjoint / correction operator in iterative methods.
    """
    sino_T = sinogram.T
    recon = iradon(sino_T, theta=angles_deg, filter_name=None,
                   output_size=img_size, circle=False)
    return recon
