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

class ModeBasis(list):
    def __init__(self, modes, grid=None):
        super().__init__(modes)
        self.grid = grid
        self._transformation_matrix = np.array(modes).T
        
    @property
    def transformation_matrix(self):
        return self._transformation_matrix
        
    def linear_combination(self, coefficients):
        return Field(self.transformation_matrix.dot(coefficients), self.grid)

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

def disk_harmonic_energy(n, m, bc='dirichlet'):
    m = abs(m)
    if bc == 'dirichlet':
        lambda_mn = scipy.special.jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        lambda_mn = scipy.special.jnp_zeros(m, n)[-1]
    return lambda_mn**2

def get_disk_harmonic_orders_sorted(num_modes, bc='dirichlet'):
    orders = [(1, 0)]
    energies = [disk_harmonic_energy(1, 0, bc)]
    results = []
    
    while len(results) < num_modes:
        k = np.argmin(energies)
        order = orders[k]
        
        if order[1] != 0:
            results.append((order[0], -order[1]))
        results.append(order)
        
        del orders[k]
        del energies[k]
        
        new_order = (order[0], order[1] + 1)
        if new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
        new_order = (order[0] + 1, order[1])
        if new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
            
    return results[:num_modes]

def make_disk_harmonic_basis(grid, num_modes, D=1, bc='dirichlet'):
    orders = get_disk_harmonic_orders_sorted(num_modes, bc)
    modes = [disk_harmonic(order[0], order[1], D, bc, grid) for order in orders]
    return ModeBasis(modes, grid)
