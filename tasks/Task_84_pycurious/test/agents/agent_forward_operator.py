import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def forward_operator(kh, beta, zt, dz, C):
    """
    Forward: Compute the theoretical radial power spectrum from
    Curie depth parameters using Bouligand et al. (2009) model.
    
    The Bouligand model computes ln(Phi) where Phi is the radial power spectrum.
    
    Parameters:
        kh: Wavenumber array (rad/km)
        beta: Fractal parameter
        zt: Top of magnetic layer (km)
        dz: Thickness of magnetic layer (km)
        C: Field constant
    
    Returns:
        log_phi: Natural log of the radial power spectrum
    """
    from pycurious import bouligand2009
    return bouligand2009(kh, beta, zt, dz, C)
