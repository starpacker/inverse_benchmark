import matplotlib

matplotlib.use('Agg')

def forward_operator(stress, C):
    """
    Compute surface displacement from stress using influence matrix.
    
    Forward Model:
        u(x,y) = C(x,y; x',y') * σ(x',y')  [linear elasticity]
        where C is the compliance/influence matrix from the 
        Boussinesq-Cerruti half-space solution.
    
    Parameters
    ----------
    stress : ndarray (n,)   Stress field [MPa].
    C : ndarray (n, n)      Influence matrix.
    
    Returns
    -------
    displacement : ndarray (n,)  Surface displacement [mm].
    """
    return C @ stress
