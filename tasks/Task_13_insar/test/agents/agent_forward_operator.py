import numpy as np
from scipy import sparse as sp

def forward_operator(F, Dx, Dy):
    """
    Compute gradients of the unwrapped phase using differentiation matrices.
    
    This implements the forward model: given an unwrapped phase F,
    compute its spatial gradients Fx (x-direction) and Fy (y-direction).
    
    Parameters
    ----------
    F : ndarray
        2D array of unwrapped phase values.
    Dx : sparse matrix
        Differentiation matrix for x-direction.
    Dy : sparse matrix
        Differentiation matrix for y-direction.
        
    Returns
    -------
    Fx : ndarray
        Gradient of F in x-direction.
    Fy : ndarray
        Gradient of F in y-direction.
    """
    rows, columns = F.shape
    Fx = (Dx @ F.ravel()).reshape(rows, columns)
    Fy = (Dy @ F.ravel()).reshape(rows, columns)
    return Fx, Fy