import matplotlib

matplotlib.use("Agg")

from skimage.transform import radon, iradon

def forward_operator(x, angles):
    """
    Forward operator: Radon transform (line integrals of attenuation).
    
    This implements the forward model for neutron tomography:
    - Physical process: Beer-Lambert law: I(θ,s) = I_0 × exp(-∫ μ(x,y) dl)
    - The sinogram p(θ,s) = -ln(I/I_0) = Radon transform of μ(x,y)
    
    Parameters
    ----------
    x : ndarray
        2D attenuation coefficient distribution μ(x,y)
    angles : ndarray
        Projection angles in degrees
        
    Returns
    -------
    y_pred : ndarray
        Predicted sinogram (Radon transform of x)
    """
    y_pred = radon(x, theta=angles, circle=False)
    return y_pred
