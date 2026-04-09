import numpy as np

import matplotlib

matplotlib.use('Agg')

def BacktrackingLineSearch(f, df, x, p, c=0.0001, rho=0.2, ls_Nite=10):
    derphi = np.real(np.dot(p.flatten(), np.conj(df(x)).flatten()))
    f0 = f(x)
    alphak = 1.0
    f_try = f(x + alphak * p)
    i = 0
    while i < ls_Nite and f_try - f0 > c * alphak * derphi and f_try > f0:
        alphak = alphak * rho
        f_try = f(x + alphak * p)
        i += 1
    return alphak, i
