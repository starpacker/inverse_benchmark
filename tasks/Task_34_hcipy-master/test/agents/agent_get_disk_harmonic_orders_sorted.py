import numpy as np

import scipy.special

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
