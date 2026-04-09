import numpy as np

def _bs_three_point(cols):
    """Deconvolution basis matrix for three_point method."""
    def I0diag(i, j):
        return np.log((np.sqrt((2*j+1)**2-4*i**2) + 2*j+1)/(2*j))/(2*np.pi)
    def I0(i, j):
        return np.log(((np.sqrt((2*j + 1)**2 - 4*i**2) + 2*j + 1)) /
                      (np.sqrt((2*j - 1)**2 - 4*i**2) + 2*j - 1))/(2*np.pi)
    def I1diag(i, j):
        return np.sqrt((2*j+1)**2 - 4*i**2)/(2*np.pi) - 2*j*I0diag(i, j)
    def I1(i, j):
        return (np.sqrt((2*j+1)**2 - 4*i**2) -
                np.sqrt((2*j-1)**2 - 4*i**2))/(2*np.pi) - 2*j*I0(i, j)

    D = np.zeros((cols, cols))
    I, J = np.diag_indices(cols)
    I, J = I[1:], J[1:]
    Ib, Jb = I, J-1
    Iu, Ju = I-1, J
    Iu, Ju = Iu[1:], Ju[1:]
    Iut, Jut = np.triu_indices(cols, k=2)
    Iut, Jut = Iut[1:], Jut[1:]

    D[Ib, Jb] = I0diag(Ib, Jb+1) - I1diag(Ib, Jb+1)
    D[I, J] = I0(I, J+1) - I1(I, J+1) + 2*I1diag(I, J)
    D[Iu, Ju] = I0(Iu, Ju+1) - I1(Iu, Ju+1) + 2*I1(Iu, Ju) - I0diag(Iu, Ju-1) - I1diag(Iu, Ju-1)
    D[Iut, Jut] = I0(Iut, Jut+1) - I1(Iut, Jut+1) + 2*I1(Iut, Jut) - I0(Iut, Jut-1) - I1(Iut, Jut-1)

    D[0, 2] = I0(0, 3) - I1(0, 3) + 2*I1(0, 2) - I0(0, 1) - I1(0, 1)
    D[0, 1] = I0(0, 2) - I1(0, 2) + 2*I1(0, 1) - 1/np.pi
    D[0, 0] = I0(0, 1) - I1(0, 1) + 1/np.pi
    return D

def run_inversion(y, dr=1.0):
    """
    Inverse Abel transform using Dasch Three-Point method.
    Maps Projection (y) -> Object (result)
    """
    IM = np.atleast_2d(y)
    rows, cols = IM.shape
    
    # Get operator matrix
    D = _bs_three_point(cols)
    
    # Perform tensordot contraction: result_ri = sum_j (IM_rj * D_ji)
    # axes=(1, 1) usually means sum over axis 1 of IM and axis 1 of D.
    # Logic check:
    # IM is (rows, cols). D is (cols, cols).
    # We want (rows, cols).
    # Dasch implementation: result = IM . D^T? 
    # The helper D indices were (j, i) where j is column index of projection.
    # The standard implementation uses tensordot(IM, D, axes=(1,1))
    
    recon = np.tensordot(IM, D, axes=(1, 1))
    
    # Scale by dr
    return recon / dr
