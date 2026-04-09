import numpy as np

import matplotlib

matplotlib.use('Agg')

class TV2d_r:
    def __init__(self):
        self.ndim = 2
    
    def grad(self, x):
        sx = x.shape[0]
        sy = x.shape[1]
        Dx = x[np.r_[1:sx, sx-1], :] - x
        self.rx = x[sx-1, :]
        Dy = x[:, np.r_[1:sy, sy-1]] - x
        self.ry = x[:, sy-1]
        res = np.zeros(x.shape + (self.ndim,), dtype=x.dtype)
        res[..., 0] = Dx
        res[..., 1] = Dy
        return res

    def adjgradx(self, x):
        sx = x.shape[0]
        x[sx-1, :] = self.rx
        x = np.flip(np.cumsum(np.flip(x, 0), 0), 0)
        return x

    def adjgrady(self, x):
        sy = x.shape[1]
        x[:, sy-1] = self.ry
        x = np.flip(np.cumsum(np.flip(x, 1), 1), 1)
        return x

    def Div(self, y):
        res = self.adjDx(y[..., 0]) + self.adjDy(y[..., 1])
        return res
    
    def adjDx(self, x):
        sx = x.shape[0]
        res = x[np.r_[0, 0:sx-1], :] - x
        res[0, :] = -x[0, :]
        res[-1, :] = x[-2, :]
        return res

    def adjDy(self, x):
        sy = x.shape[1]
        res = x[:, np.r_[0, 0:sy-1]] - x
        res[:, 0] = -x[:, 0]
        res[:, -1] = x[:, -2]
        return res

    def amp(self, grad):
        amp = np.sqrt(np.sum(grad ** 2, axis=(len(grad.shape)-1)))
        amp_shape = amp.shape + (1,)
        d = np.ones(amp.shape + (self.ndim,), dtype=amp.dtype)
        d = np.multiply(amp.reshape(amp_shape), d)
        return d

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

def prox_tv2d_r(y, lambda_tv, step=0.1):
    sizeg = y.shape + (2,)
    G = np.zeros(sizeg)
    i = 0
    tvopt = TV2d_r()
    while i < 40:
        dG = tvopt.grad(tvopt.Div(G) - y/lambda_tv)
        G = G - step*dG
        d = tvopt.amp(G)
        G = G/np.maximum(d, 1.0*np.ones(sizeg))
        i = i + 1
    f = y - lambda_tv * tvopt.Div(G)
    return f

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

def ADMM_l2Afxnb_tvx(Afunc, invAfunc, b, Nite, step, tv_r, rho, cgd_Nite=3, tvndim=2):
    z = invAfunc(b)
    u = np.zeros(z.shape)
    tvprox = prox_tv2d_r
    for i in range(Nite):
        x = prox_l2_Afxnb_CGD(Afunc, invAfunc, b, z-u, rho, cgd_Nite)
        z = tvprox(x + u, 2.0 * tv_r/rho)
        u = u + step * (x - z)
        print(f'Iteration {i}, gradient in ADMM {np.linalg.norm(x-z):.4f}')
    return x
