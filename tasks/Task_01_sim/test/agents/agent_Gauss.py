import math
import numpy as np
from numpy import zeros, log


# --- Extracted Dependencies ---

def Gauss(sigma):
    sigma = np.array(sigma, dtype='float32')
    s = sigma.size
    if s == 1:
        sigma = [sigma, sigma]
    sigma = np.array(sigma, dtype='float32')
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * log(2)))
    dim = len(N)
    if dim > 1:
        N[1] = np.maximum(N[0], N[1])
        N[0] = N[1]
    if dim == 1:
        x = np.arange(-np.fix(N / 2), np.ceil(N / 2), dtype='float32')
        PSF = np.exp(-0.5 * (x * x) / (np.dot(sigma, sigma)))
        PSF = PSF / PSF.sum()
        return PSF
    if dim == 2:
        m = N[0]
        n = N[1]
        x = np.arange(-np.fix((n / 2)), np.ceil((n / 2)), dtype='float32')
        y = np.arange(-np.fix((m / 2)), np.ceil((m / 2)), dtype='float32')
        X, Y = np.meshgrid(x, y)
        s1 = sigma[0]
        s2 = sigma[1]
        PSF = np.exp(-(X * X) / (2 * np.dot(s1, s1)) - (Y * Y) / (2 * np.dot(s2, s2)))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        return PSF
    if dim == 3:
        m = N[0]
        n = N[1]
        k = N[2]
        x = np.arange(-np.fix(n / 2), np.ceil(n / 2), dtype='float32')
        y = np.arange(-np.fix(m / 2), np.ceil(m / 2), dtype='float32')
        z = np.arange(-np.fix(k / 2), np.ceil(k / 2), dtype='float32')
        [X, Y, Z] = np.meshgrid(x, y, z)
        s1 = sigma[0]
        s2 = sigma[1]
        s3 = sigma[2]
        PSF = np.exp(-(X * X) / (2 * s1 * s1) - (Y * Y) / (2 * s2 * s2) - (Z * Z) / (2 * s3 ** 2))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        return PSF
