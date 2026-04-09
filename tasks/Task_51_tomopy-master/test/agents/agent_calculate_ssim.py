def calculate_ssim(gt, recon):
    """Structural Similarity Index Wrapper"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt.max() - gt.min()
        return structural_similarity(gt, recon, data_range=data_range)
    except ImportError:
        return 0
