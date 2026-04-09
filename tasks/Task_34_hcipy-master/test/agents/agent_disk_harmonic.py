import numpy as np

import scipy.special

class Field(np.ndarray):
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        
    @property
    def shaped(self):
        if self.grid.dims:
            return self.reshape(self.grid.dims)
        return self

def disk_harmonic(n, m, D=1, bc='dirichlet', grid=None):
    x = grid.x
    y = grid.y
    r = np.sqrt(x**2 + y**2) * 2 / D
    theta = np.arctan2(y, x)
    
    m_negative = m < 0
    m = abs(m)
    
    if bc == 'dirichlet':
        lambda_mn = scipy.special.jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        lambda_mn = scipy.special.jnp_zeros(m, n)[-1]
    
    if m_negative:
        z = scipy.special.jv(m, lambda_mn * r) * np.sin(m * theta)
    else:
        z = scipy.special.jv(m, lambda_mn * r) * np.cos(m * theta)
        
    mask = r <= 1
    z = z * mask
    norm = np.sqrt(np.sum(z[mask]**2))
    if norm > 0: z /= norm
    
    return Field(z, grid)
