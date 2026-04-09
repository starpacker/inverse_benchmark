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

def prox_l2_Afxnb_CGD(Afunc, invAfunc, b, x0, rho, Nite, ls_Nite=10):
    eps = 0.001
    i = 0
    def f(xi):
        return np.linalg.norm(Afunc(xi)-b)**2 + (rho/2)*np.linalg.norm(xi-x0)**2
    def df(xi):
        return 2*invAfunc(Afunc(xi)-b) + rho*(xi-x0)
    dx = -df(x0)
    alpha, nstp = BacktrackingLineSearch(f, df, x0, dx, ls_Nite=ls_Nite)
    x = x0 + alpha * dx
    s = dx
    delta0 = np.linalg.norm(dx)
    deltanew = delta0
    while i < Nite and deltanew > eps*delta0 and nstp < ls_Nite:
        dx = -df(x)
        deltaold = deltanew
        deltanew = np.linalg.norm(dx)
        if deltaold == 0:
            beta = 0
        else:
            beta = float(deltanew / float(deltaold))
        s = dx + beta * s
        alpha, nstp = BacktrackingLineSearch(f, df, x, s, ls_Nite=ls_Nite)
        x = x + alpha * s
        i = i + 1
    return x
