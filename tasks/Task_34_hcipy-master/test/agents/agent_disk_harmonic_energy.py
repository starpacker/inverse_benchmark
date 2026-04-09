import scipy.special

def disk_harmonic_energy(n, m, bc='dirichlet'):
    m = abs(m)
    if bc == 'dirichlet':
        lambda_mn = scipy.special.jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        lambda_mn = scipy.special.jnp_zeros(m, n)[-1]
    return lambda_mn**2
