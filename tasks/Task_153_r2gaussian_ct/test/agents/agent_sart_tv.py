import matplotlib

matplotlib.use('Agg')

import numpy as np

from skimage.transform import radon, iradon

def sart_tv(sinogram, theta, size, n_outer=25, n_tv=8, lam_tv=0.15, relax=0.08):
    """SART + TV regularization iterative reconstruction."""
    x = iradon(sinogram, theta=theta, filter_name='ramp')[:size, :size]
    x = np.clip(x, 0, None)
    n_angles = len(theta)

    for outer in range(n_outer):
        cur_relax = relax * (1.0 - 0.5 * outer / n_outer)
        angle_order = np.random.RandomState(outer).permutation(n_angles)
        for j in angle_order:
            th = np.array([theta[j]])
            proj_est = radon(x, theta=th)
            mr = min(proj_est.shape[0], sinogram.shape[0])
            res = np.zeros_like(proj_est)
            res[:mr, 0] = sinogram[:mr, j] - proj_est[:mr, 0]
            bp = iradon(res, theta=th, filter_name=None)[:size, :size]
            x = x + cur_relax * bp
            x = np.clip(x, 0, None)

        cur_tv = lam_tv * (1.0 - 0.3 * outer / n_outer)
        for _ in range(n_tv):
            gx = np.diff(x, axis=1, append=x[:, -1:])
            gy = np.diff(x, axis=0, append=x[-1:, :])
            norm = np.sqrt(gx**2 + gy**2 + 1e-8)
            gx /= norm
            gy /= norm
            div = (np.diff(gx, axis=1, prepend=gx[:, :1])
                   + np.diff(gy, axis=0, prepend=gy[:1, :]))
            x = x + cur_tv * div
            x = np.clip(x, 0, None)

        if (outer + 1) % 5 == 0:
            print(f"    SART-TV outer {outer+1}/{n_outer}")

    return x
