import math
import warnings
import numpy as np
import gc
import pywt
import time
from numpy import zeros, log


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


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


def rliter(yk, data, otf):
    rliter_val = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    return rliter_val


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


def Low_frequency_resolve(coeffs, dlevel):
    cAn = coeffs[0]
    vec = []
    vec.append(cAn)
    for i in range(1, dlevel + 1):
        (cH, cV, cD) = coeffs[i]
        [cH_x, cH_y] = cH.shape
        cH_new = np.zeros((cH_x, cH_y))
        t = (cH_new, cH_new, cH_new)
        vec.append(t)
    return vec


def rm_1(Biter, x, y):
    Biter_new = np.zeros((x, y), dtype=('uint8'))
    if x % 2 and y % 2 == 0:
        Biter_new[:, :] = Biter[0:x, :]
    elif x % 2 == 0 and y % 2:
        Biter_new[:, :] = Biter[:, 0:y]
    elif x % 2 and y % 2:
        Biter_new[:, :] = Biter[0:x, 0:y]
    else:
        Biter_new = Biter
    return Biter_new


def background_estimation(imgs, th=1, dlevel=7, wavename='db6', iter=3):
    try:
        [t, x, y] = imgs.shape
        Background = np.zeros((t, x, y))
        for taxial in range(t):
            img = imgs[taxial, :, :]
            for i in range(iter):
                initial = img
                res = initial
                coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                vec = Low_frequency_resolve(coeffs, dlevel)
                Biter = pywt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)
                if th > 0:
                    eps = np.sqrt(np.abs(res)) / 2
                    ind = initial > (Biter_new + eps)
                    res[ind] = Biter_new[ind] + eps[ind]
                    coeffs1 = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                    vec = Low_frequency_resolve(coeffs1, dlevel)
                    Biter = pywt.waverec2(vec, wavelet=wavename)
                    Biter_new = rm_1(Biter, x, y)
                    Background[taxial, :, :] = Biter_new
    except ValueError:
        [x, y] = imgs.shape
        Background = np.zeros((x, y))
        for i in range(iter):
            initial = imgs
            res = initial
            coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
            vec = Low_frequency_resolve(coeffs, dlevel)
            Biter = pywt.waverec2(vec, wavelet=wavename)
            Biter_new = rm_1(Biter, x, y)
            if th > 0:
                eps = np.sqrt(np.abs(res)) / 2
                ind = initial > (Biter_new + eps)
                res[ind] = Biter_new[ind] + eps[ind]
                coeffs1 = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                vec = Low_frequency_resolve(coeffs1, dlevel)
                Biter = pywt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)
                Background = Biter_new
    return Background


def spatial_upsample(SIMmovie, n=2):
    k = SIMmovie.ndim
    if k > 2:
        [sz, sx, sy] = SIMmovie.shape
        y = np.zeros((sz, sx * n, sy * n), dtype='float32')
        for frames in range(0, sz):
            y[frames, 0:sx * n:n, 0:sy * n:n] = SIMmovie[frames, :, :]
        return y
    else:
        [sx, sy] = SIMmovie.shape
        y = np.zeros((sx * n, sy * n), dtype='float32')
        y[0:sx * n:n, 0:sy * n:n] = SIMmovie
        return y


def fInterp_2D(img, newsz):
    imgsz = img.shape
    imgsz = np.array(imgsz)
    newsz = np.array(newsz)
    if (np.sum(newsz == 0)) >= 1:
        img_ip = []
        return img_ip
    isgreater = newsz >= imgsz
    isgreater = isgreater.astype(int)
    isgreater = np.array(isgreater)
    incr = np.zeros((2, 1), dtype='float32')
    for iDim in range(0, 2):
        if isgreater[0][iDim] == 1:
            incr[iDim] = 1
        else:
            incr[iDim] = np.floor(imgsz[iDim] / newsz[iDim]) + 1
    newsz[0][0] = int(newsz[0][0])
    a = int(newsz[0][0])
    b = int(newsz[0][1])
    nyqst = np.ceil((imgsz + 1) / 2)
    B = float(a / imgsz[0] * b / imgsz[1])
    img = B * np.fft.fft2(img)
    img_ip = np.zeros((a, b), dtype='complex')
    img_ip[0: int(nyqst[0]), 0: int(nyqst[1])] = img[0: int(nyqst[0]), 0: int(nyqst[1])]
    img_ip[a - (int(imgsz[0]) - int(nyqst[0])):a, 0:int(nyqst[1])] = img[int(nyqst[0]):int(imgsz[0]), 0:int(nyqst[1])]
    img_ip[0: int(nyqst[0]), a - (int(imgsz[1]) - int(nyqst[1])):a] = img[0: int(nyqst[0]), int(nyqst[1]): int(imgsz[1])]
    img_ip[a - (int(imgsz[0]) - int(nyqst[0])):a, a - (int(imgsz[1]) - int(nyqst[1])):a] = img[int(nyqst[0]):int(imgsz[0]), int(nyqst[1]):int(imgsz[1])]
    rm = np.remainder(imgsz, 2)
    if int(rm[0]) == 0 and int(a) != int(imgsz[0]):
        img_ip[int(nyqst[0]), :] = img_ip[int(nyqst[0]), :] / 2
        img_ip[int(nyqst[0]) + int(a) - int(imgsz[0]), :] = img_ip[int(nyqst[0]), :]
    if int(rm[1]) == 0 and int(b) != int(imgsz[1]):
        img_ip[:, int(nyqst[1])] = img_ip[:, int(nyqst[1])] / 2
        img_ip[:, int(nyqst[1]) + int(b) - int(imgsz[1])] = img_ip[:, int(nyqst[1])]
    img_ip = np.array(img_ip)
    img_ip = (np.fft.ifft2(img_ip)).real
    img_ip = img_ip[0: int(a):int(incr[0]), 0:int(b): int(incr[1])]
    return img_ip


def fourier_upsample(imgstack, n=2):
    n = n * np.ones((1, 2))
    if imgstack.ndim < 3:
        z = 1
        sz = [imgstack.shape[0], imgstack.shape[1]]
        imgfl = np.zeros((int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))
    else:
        z = imgstack.shape[0]
        sz = [imgstack.shape[1], imgstack.shape[2]]
        imgfl = np.zeros((z, int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))

    for i in range(0, z):
        if imgstack.ndim < 3:
            img = imgstack
        else:
            img = imgstack[i, :, :]
        imgsz = [img.shape[0], img.shape[1]]
        imgsz = np.array(imgsz)
        if ((imgsz[0] % 2)) == 1:
            sz_local = imgsz
        else:
            sz_local = imgsz - 1
        sz_local = np.array(sz_local)
        idx = np.ceil(sz_local / 2) + 1 + (n - 1) * np.floor(sz_local / 2)
        padsize = [img.shape[0] / 2, img.shape[1] / 2]
        padsize = np.array(padsize)
        k = np.ceil(padsize)
        f_pad = np.floor(padsize)

        img = np.pad(img, ((int(k[0]), 0), (int(k[1]), 0)), 'symmetric')
        img = np.pad(img, ((0, int(f_pad[0])), (0, int(f_pad[1]))), 'symmetric')

        im_shape = n * (np.array(img.shape))
        newsz = np.floor(im_shape - (n - 1))
        imgl = fInterp_2D(img, newsz)
        if imgstack.ndim < 3:
            imgfl = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0]) + int(idx[0][0]),
                    int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) * int(imgsz[1])]
        else:
            imgfl = np.array(imgfl)
            imgfl[i, :, :] = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0]) + int(idx[0][0]),
                             int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) * int(imgsz[1])]

    return imgfl


# ============================================================================
# COMPONENT 1: load_and_preprocess_data
# ============================================================================

def load_and_preprocess_data(input_path, up_sample=0):
    """
    Load image from file and preprocess it.
    
    Parameters
    ----------
    input_path : str
        Path to input image file.
    up_sample : int
        0: No upsampling
        1: Fourier upsampling
        2: Spatial upsampling
    
    Returns
    -------
    img_preprocessed : ndarray
        Preprocessed image ready for inversion.
    scaler : float
        Original maximum value for rescaling output.
    """
    from skimage import io
    
    img = io.imread(input_path)
    img = np.array(img, dtype='float32')
    
    scaler = np.max(img)
    img = img / scaler
    
    backgrounds = background_estimation(img / 2.5)
    img = img - backgrounds
    
    img = img / (img.max())
    img[img < 0] = 0
    
    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)
    
    img = img / (img.max())
    
    return img, scaler


# ============================================================================
# COMPONENT 2: forward_operator
# ============================================================================

def forward_operator(x, kernel):
    """
    Apply the forward imaging model (convolution with PSF).
    
    Parameters
    ----------
    x : ndarray
        Input image (latent/reconstructed image).
    kernel : ndarray
        Point spread function kernel.
    
    Returns
    -------
    y_pred : ndarray
        Predicted observation after applying PSF convolution.
    """
    if x.ndim > 2:
        y_pred = np.zeros_like(x, dtype='float32')
        for i in range(x.shape[0]):
            otf = psf2otf(kernel, x[i].shape)
            y_pred[i] = np.fft.ifftn(otf * np.fft.fftn(x[i])).real
    else:
        otf = psf2otf(kernel, x.shape)
        y_pred = np.fft.ifftn(otf * np.fft.fftn(x)).real
    
    return y_pred.astype('float32')


# ============================================================================
# COMPONENT 3: run_inversion
# ============================================================================

def run_inversion(img, sigma, sparse_iter=100, fidelity=150, sparsity=10, tcontinuity=0.5,
                  deconv_iter=7, deconv_type=1, mu=1):
    """
    Run the sparse deconvolution inversion algorithm.
    
    Parameters
    ----------
    img : ndarray
        Preprocessed input image.
    sigma : float or list
        PSF sigma parameter(s).
    sparse_iter : int
        Number of sparse Hessian iterations.
    fidelity : int
        Fidelity parameter.
    sparsity : int
        Sparsity parameter.
    tcontinuity : float
        Continuity along z-axis.
    deconv_iter : int
        Number of deconvolution iterations.
    deconv_type : int
        0: No deconvolution
        1: Richardson-Lucy
        2: LandWeber
    mu : float
        Regularization parameter.
    
    Returns
    -------
    img_result : ndarray
        Reconstructed image (normalized).
    """
    # -------------------------------------------------------------------------
    # Sparse Hessian Reconstruction
    # -------------------------------------------------------------------------
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
    del xxfft, yyfft, zzfft, xyfft, xzfft, yzfft, operationfft
    gc.collect()
    
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
    
    start = time.process_time()
    
    for iter_idx in range(0, sparse_iter):
        g_update = np.fft.fftn(g_update)
        
        if iter_idx == 0:
            g = np.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = np.fft.ifftn(np.divide(g_update, normlize)).real
        
        g_update = np.multiply((fidelity / mu), f)
        
        Lxx, bxx = iter_xx(g, bxx, 1, mu)
        g_update = g_update + Lxx
        del Lxx
        gc.collect()
        
        Lyy, byy = iter_yy(g, byy, 1, mu)
        g_update = g_update + Lyy
        del Lyy
        gc.collect()
        
        Lzz, bzz = iter_zz(g, bzz, contiz ** 2, mu)
        g_update = g_update + Lzz
        del Lzz
        gc.collect()
        
        Lxy, bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy
        del Lxy
        gc.collect()
        
        Lxz, bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update = g_update + Lxz
        del Lxz
        gc.collect()
        
        Lyz, byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update = g_update + Lyz
        del Lyz
        gc.collect()
        
        Lsparse, bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update = g_update + Lsparse
        del Lsparse
        gc.collect()
        
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
    
    del bxx, byy, bzz, bxy, byz, bl1, f, normlize, g_update
    gc.collect()
    
    img_sparse = g[1, :, :] if flage else g
    
    end = time.process_time()
    print('sparse-hessian time %0.2fs' % (end - start))
    
    img_sparse = img_sparse / (img_sparse.max())
    
    # -------------------------------------------------------------------------
    # Iterative Deconvolution
    # -------------------------------------------------------------------------
    if deconv_type == 0 or not sigma:
        img_result = img_sparse
        return img_result
    
    start = time.process_time()
    kernel = Gauss(sigma)
    
    # Core deconvolution
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
    
    end = time.process_time()
    print('deconv time %0.2fs' % (end - start))
    
    return img_result


def _deblur_core(data, kernel, iteration, rule):
    """Internal deblur core function."""
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
        # LandWeber deconv
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
        # Richardson-Lucy deconv
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


# ============================================================================
# COMPONENT 4: evaluate_results
# ============================================================================

def evaluate_results(img_recon, expected_output_path, output_path, scaler, original_dtype):
    """
    Evaluate reconstruction results and save output.
    
    Parameters
    ----------
    img_recon : ndarray
        Reconstructed image (normalized).
    expected_output_path : str
        Path to expected output for comparison.
    output_path : str
        Path to save reconstructed image.
    scaler : float
        Scaling factor to restore original intensity range.
    original_dtype : dtype
        Original image data type for saving.
    
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, and MSE values.
    """
    from skimage import io
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
    
    # Rescale output
    img_output = scaler * img_recon
    
    # Save result
    io.imsave(output_path, img_output.astype(original_dtype))
    print(f"✅ Processing complete! Result saved to: {output_path}")
    
    # Load expected output
    expected = io.imread(expected_output_path)
    
    # Ensure shape matches
    if img_output.shape != expected.shape:
        raise ValueError("Reconstructed image and expected image must have the same shape!")
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(expected, img_output, data_range=expected.max() - expected.min())
    ssim_val = ssim(expected, img_output, data_range=expected.max() - expected.min(),
                   channel_axis=None if len(expected.shape) == 2 else 2)
    mse = np.mean((expected.astype(np.float64) - img_output.astype(np.float64)) ** 2)
    
    print(f"📊 PSNR: {psnr:.4f} dB")
    print(f"📊 SSIM: {ssim_val:.4f}")
    print(f"📊 MSE: {mse:.6f}")
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim_val,
        'mse': mse
    }
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    from skimage import io
    
    # Configuration parameters
    PSF_FACTOR = 280 / 65
    SPARSE_ITER = 1000
    
    # File paths
    input_path = "001.tif"
    output_path = "output.tif"
    expected_output_path = "expected_output.tif"
    
    # Get original dtype
    im_original = io.imread(input_path)
    original_dtype = im_original.dtype
    
    # Step 1: Load and preprocess data
    img_preprocessed, scaler = load_and_preprocess_data(input_path, up_sample=0)
    
    # Step 2: Run inversion (includes sparse Hessian and iterative deconvolution)
    img_recon = run_inversion(
        img_preprocessed,
        sigma=PSF_FACTOR,
        sparse_iter=SPARSE_ITER,
        fidelity=150,
        sparsity=10,
        tcontinuity=0.5,
        deconv_iter=7,
        deconv_type=1,
        mu=1
    )
    
    # Step 3: Evaluate results
    metrics = evaluate_results(img_recon, expected_output_path, output_path, scaler, original_dtype)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")