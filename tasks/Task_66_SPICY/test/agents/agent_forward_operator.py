import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(u, v, dx, dy, rho):
    """
    Forward operator: Compute the RHS of the pressure Poisson equation (PPE)
    from velocity fields.
    
    ∇²p = -ρ (du/dx·du/dx + 2·du/dy·dv/dx + dv/dy·dv/dy)
    
    Parameters:
        u, v: Velocity field components
        dx, dy: Grid spacing
        rho: Fluid density
    
    Returns:
        rhs: Right-hand side of the pressure Poisson equation
    """
    # Compute velocity gradients using central differences
    dudx = np.gradient(u, dx, axis=0)
    dudy = np.gradient(u, dy, axis=1)
    dvdx = np.gradient(v, dx, axis=0)
    dvdy = np.gradient(v, dy, axis=1)
    
    # PPE RHS
    rhs = -rho * (dudx**2 + 2 * dudy * dvdx + dvdy**2)
    
    return rhs
