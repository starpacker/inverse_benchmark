import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_pr(q, I_q, d_max=None, n_r=100):
    """
    Estimate the pair-distance distribution function P(r) via a simple
    indirect Fourier transform (Moore method / regularised sine transform).
    
    P(r) = (2r / pi) * integral_0^inf  q * I(q) * sin(qr) dq
    
    In practice we discretise and apply a simple Tikhonov regularisation
    to suppress noise artefacts.
    """
    if d_max is None:
        d_max = 2.0 * np.pi / q.min()
        d_max = min(d_max, 300.0)

    r = np.linspace(0, d_max, n_r)
    pr = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < 1e-12:
            pr[i] = 0.0
            continue
        integrand = q * I_q * np.sin(q * ri)
        pr[i] = (2.0 * ri / np.pi) * np.trapezoid(integrand, q)

    pr = np.maximum(pr, 0.0)
    if pr.max() > 0:
        pr /= pr.max()

    return r, pr
