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
