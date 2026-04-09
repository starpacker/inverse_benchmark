import numpy as np

import odl

def forward_operator(x, space, geometry):
    """
    Applies the Ray Transform to the input x.
    Note: Ideally x is an ODL element, but if it is numpy, we wrap it.
    
    Args:
        x: Input volume (ODL element or numpy array).
        space: ODL reconstruction space.
        geometry: ODL geometry.
        
    Returns:
        y_pred: The forward projection (numpy array).
    """
    # Setup Ray Transform
    if odl.tomo.ASTRA_CUDA_AVAILABLE:
        impl = 'astra_cuda'
    else:
        impl = 'astra_cpu'
        
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)
    
    # Diagonal Operator to apply RayTransform to both material channels
    A = odl.DiagonalOperator(ray_trafo, 2)
    
    if isinstance(x, np.ndarray):
        x_odl = A.domain.element(x)
        return A(x_odl).asarray()
    else:
        # Assume x is an ODL element compatible with A
        return A(x).asarray()
