import numpy as np

import numpy.typing as npt

def forward_operator(x_model: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Computes the forward projection (Prediction) for a given 3D point.
    Not strictly used in the inverse fitting loop provided (which uses backward projection logic),
    but essential for validation and completeness.
    x_model: [X, Y, Z, U, V, rho_scaling, na, n]
    Returns: [x_sensor, y_sensor]
    """
    X, Y, Z, u, v, rho_scaling, na, n = x_model
    
    rho = np.sqrt(u**2 + v**2)
    dr_sq = 1 - rho * (na / n)**2
    if dr_sq < 0:
        return np.array([np.nan, np.nan])
        
    phi = -(na / n) / np.sqrt(dr_sq)
    alpha_u = u * phi
    alpha_v = v * phi
    
    u_micron = u / rho_scaling
    v_micron = v / rho_scaling
    
    x_sensor = X + Z * alpha_u + u_micron
    y_sensor = Y + Z * alpha_v + v_micron
    
    return np.array([x_sensor, y_sensor])
