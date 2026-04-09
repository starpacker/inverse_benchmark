import matplotlib

matplotlib.use('Agg')

from skimage.metrics import structural_similarity as ssim

def compute_ssim(gt, recon):
    """Structural similarity index."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)
