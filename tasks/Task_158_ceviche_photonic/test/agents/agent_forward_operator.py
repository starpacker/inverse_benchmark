import matplotlib

matplotlib.use('Agg')

from ceviche import fdfd_ez

def forward_operator(eps_r, source, omega, dl, npml):
    """
    FDFD forward solve: given permittivity distribution, compute Ez field.
    
    Args:
        eps_r: 2D permittivity distribution
        source: 2D source distribution
        omega: angular frequency
        dl: grid spacing
        npml: PML thickness
    
    Returns:
        Ez: complex electric field
    """
    F = fdfd_ez(omega, dl, eps_r, [npml, npml])
    _, _, Ez = F.solve(source)
    return Ez
