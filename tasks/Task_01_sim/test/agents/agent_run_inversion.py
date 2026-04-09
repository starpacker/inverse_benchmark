import numpy as np
import math
from numpy import zeros, log

# --- 1. Frequency Domain Helpers ---

def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = np.array(outSize - psfSize)
    psf = np.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

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

# --- 2. Finite Difference Operators ---

def forward_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])]
    return -out

def back_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out

def operation_xx(gsize):
    delta_xx = np.array([[[1, -2, 1]]], dtype='float32')
    xxfft = np.fft.fftn(delta_xx, gsize) * np.conj(np.fft.fftn(delta_xx, gsize))
    return xxfft

def operation_xy(gsize):
    delta_xy = np.array([[[1, -1], [-1, 1]]], dtype='float32')
    xyfft = np.fft.fftn(delta_xy, gsize) * np.conj(np.fft.fftn(delta_xy, gsize))
    return xyfft

def operation_xz(gsize):
    delta_xz = np.array([[[1, -1]], [[-1, 1]]], dtype='float32')
    xzfft = np.fft.fftn(delta_xz, gsize) * np.conj(np.fft.fftn(delta_xz, gsize))
    return xzfft

def operation_yy(gsize):
    delta_yy = np.array([[[1], [-2], [1]]], dtype='float32')
    yyfft = np.fft.fftn(delta_yy, gsize) * np.conj(np.fft.fftn(delta_yy, gsize))
    return yyfft

def operation_yz(gsize):
    delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
    yzfft = np.fft.fftn(delta_yz, gsize) * np.conj(np.fft.fftn(delta_yz, gsize))
    return yzfft

def operation_zz(gsize):
    delta_zz = np.array([[[1]], [[-2]], [[1]]], dtype='float32')
    zzfft = np.fft.fftn(delta_zz, gsize) * np.conj(np.fft.fftn(delta_zz, gsize))
    return zzfft

# --- 3. Split Bregman Iterators ---

def shrink(x, L):
    s = np.abs(x)
    xs = np.sign(x) * np.maximum(s - 1 / L, 0)
    return xs

def iter_xx(g, bxx, para, mu):
    gxx = back_diff(forward_diff(g, 1, 1), 1, 1)
    dxx = shrink(gxx + bxx, mu)
    bxx = bxx + (gxx - dxx)
    Lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1), 1, 1)
    return Lxx, bxx

def iter_xy(g, bxy, para, mu):
    gxy = forward_diff(forward_diff(g, 1, 1), 1, 2)
    dxy = shrink(gxy + bxy, mu)
    bxy = bxy + (gxy - dxy)
    Lxy = para * back_diff(back_diff(dxy - bxy, 1, 2), 1, 1)
    return Lxy, bxy

def iter_xz(g, bxz, para, mu):
    gxz = forward_diff(forward_diff(g, 1, 1), 1, 0)
    dxz = shrink(gxz + bxz, mu)
    bxz = bxz + (gxz - dxz)
    Lxz = para * back_diff(back_diff(dxz - bxz, 1, 0), 1, 1)
    return Lxz, bxz

def iter_yy(g, byy, para, mu):
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy

def iter_yz(g, byz, para, mu):
    gyz = forward_diff(forward_diff(g, 1, 2), 1, 0)
    dyz = shrink(gyz + byz, mu)
    byz = byz + (gyz - dyz)
    Lyz = para * back_diff(back_diff(dyz - byz, 1, 0), 1, 2)
    return Lyz, byz

def iter_zz(g, bzz, para, mu):
    gzz = back_diff(forward_diff(g, 1, 0), 1, 0)
    dzz = shrink(gzz + bzz, mu)
    bzz = bzz + (gzz - dzz)
    Lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0), 1, 0)
    return Lzz, bzz

def iter_sparse(gsparse, bsparse, para, mu):
    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse

# --- 4. Main Driver ---

def run_inversion(img, sigma, sparse_iter=100, fidelity=150, sparsity=10, tcontinuity=0.5,
                  deconv_iter=7, deconv_type=1, mu=1):
    contiz = np.sqrt(tcontinuity)
    f1 = img
    flage = 0
    
    f_flag = img.ndim
    if f_flag == 2:
        contiz = 0
        flage = 1
        f = np.zeros((3, img.shape[0], img.shape[1]), dtype='float32')
        for i in range(0, 3):
            f[i, :, :] = f1
    elif f_flag > 2:
        if f1.shape[0] < 3:
            contiz = 0
            f = np.zeros((3, img.shape[1], img.shape[2]), dtype='float32')
            f[0:f1.shape[0], :, :] = f1
            for i in range(f1.shape[0], 3):
                f[i, :, :] = f[1, :, :]
        else:
            f = f1
    else:
        f = f1
    
    imgsize = np.shape(f)
    
    print("Start the Sparse deconvolution...")
    
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)
    
    operationfft = xxfft + yyfft + (contiz ** 2) * zzfft + 2 * xyfft + 2 * (contiz) * xzfft + 2 * (contiz) * yzfft
    normlize = (fidelity / mu) + (sparsity ** 2) + operationfft
    
    bxx = np.zeros(imgsize, dtype='float32')
    byy = bxx.copy()
    bzz = bxx.copy()
    bxy = bxx.copy()
    bxz = bxx.copy()
    byz = bxx.copy()
    bl1 = bxx.copy()
    
    g_update = np.multiply(fidelity / mu, f)
    
    tol = 1e-4
    residual_prev = np.inf
    
    for iter_idx in range(0, sparse_iter):
        g_update = np.fft.fftn(g_update)
        
        if iter_idx == 0:
            g = np.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = np.fft.ifftn(np.divide(g_update, normlize)).real
        
        g_update = np.multiply((fidelity / mu), f)
        
        Lxx, bxx = iter_xx(g, bxx, 1, mu)
        g_update = g_update + Lxx
        
        Lyy, byy = iter_yy(g, byy, 1, mu)
        g_update = g_update + Lyy
        
        Lzz, bzz = iter_zz(g, bzz, contiz ** 2, mu)
        g_update = g_update + Lzz
        
        Lxy, bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy
        
        Lxz, bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update = g_update + Lxz
        
        Lyz, byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update = g_update + Lyz
        
        Lsparse, bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update = g_update + Lsparse
        
        if iter_idx % 20 == 0:
            residual = np.linalg.norm(f - g)
            rel_change = abs(residual - residual_prev) / (residual_prev + 1e-12)
            if rel_change < tol:
                print(f"Converged at iteration {iter_idx}: residual change = {rel_change:.2e}")
                break
            residual_prev = residual
        
        if iter_idx % 20 == 0:
            print('%d iterations done\r' % iter_idx)
    
    g[g < 0] = 0
    
    img_sparse = g[1, :, :] if flage else g
    
    img_sparse = img_sparse / (img_sparse.max())
    
    if deconv_type == 0 or not sigma:
        img_result = img_sparse
        return img_result
    
    kernel = Gauss(sigma)
    
    data = img_sparse
    iteration = deconv_iter
    rule = deconv_type
    
    if data.ndim > 2:
        data_de = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype='float32')
        for i in range(0, data.shape[2]):
            data_de[:, :, i] = _deblur_core(data[:, :, i], kernel, iteration, rule).real
    else:
        data_de = _deblur_core(data, kernel, iteration, rule).real
    
    img_result = data_de
    
    return img_result

def _deblur_core(data, kernel, iteration, rule):
    kernel = np.array(kernel)
    kernel = kernel / sum(sum(kernel))
    kernel_initial = kernel
    [dx, dy] = data.shape
    
    B = math.floor(min(dx, dy) / 6)
    data = np.pad(data, [int(B), int(B)], 'edge')
    yk = data
    xk = zeros((data.shape[0], data.shape[1]), dtype='float32')
    vk = zeros((data.shape[0], data.shape[1]), dtype='float32')
    otf = psf2otf(kernel_initial, data.shape)
    
    if rule == 2:
        t = 1
        gamma1 = 1
        for i in range(0, iteration):
            if i == 0:
                xk_update = data
                xk = data + t * np.fft.ifftn(np.conj(otf)) * (np.fft.fftn(data) - (otf * np.fft.fftn(data)))
            else:
                gamma2 = 1 / 2 * (4 * gamma1 * gamma1 + gamma1 ** 4) ** (1 / 2) - gamma1 ** 2
                beta = -gamma2 * (1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * np.fft.ifftn(np.conj(otf) * (np.fft.fftn(data) - (otf * np.fft.fftn(yk_update))))
                yk = np.maximum(yk, 1e-6).astype('float32')
                gamma1 = gamma2
                xk_update = xk
                xk = yk
    
    elif rule == 1:
        for iter_idx in range(0, iteration):
            xk_update = xk
            rliter1 = rliter(yk, data, otf)
            
            xk = yk * ((np.fft.ifftn(np.conj(otf) * rliter1)).real) / (
                (np.fft.ifftn(np.fft.fftn(np.ones(data.shape)) * otf)).real)
            
            xk = np.maximum(xk, 1e-6).astype('float32')
            
            vk_update = vk
            vk = np.maximum(xk - yk, 1e-6).astype('float32')
            
            if iter_idx == 0:
                alpha = 0
                yk = xk
                yk = np.maximum(yk, 1e-6).astype('float32')
                yk = np.array(yk)
            else:
                alpha = sum(sum(vk_update * vk)) / (sum(sum(vk_update * vk_update)) + 1e-10)
                alpha = np.maximum(np.minimum(alpha, 1), 1e-6).astype('float32')
                yk = xk + alpha * (xk - xk_update)
                yk = np.maximum(yk, 1e-6).astype('float32')
                yk[np.isnan(yk)] = 1e-6
    
    yk[yk < 0] = 0
    yk = np.array(yk, dtype='float32')
    data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]
    
    return data_decon

def rliter(yk, data, otf):
    rliter_val = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    return rliter_val