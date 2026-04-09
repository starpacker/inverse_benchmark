import numpy as np

import scipy.io as sio

import matplotlib

matplotlib.use('Agg')

def dim_match(A_shape, B_shape):
    A_out_shape = A_shape
    B_out_shape = B_shape
    if len(A_shape) < len(B_shape):
        for _ in range(len(A_shape), len(B_shape)):
            A_out_shape += (1,)
    elif len(A_shape) > len(B_shape):
        for _ in range(len(B_shape), len(A_shape)):
            B_out_shape += (1,)
    return A_out_shape, B_out_shape

def hankelnd_r(a, win_shape, win_strides=None):
    if win_strides is None:
        win_strides = np.ones(win_shape.__len__()).astype(int)
    win_shape = np.array(win_shape)
    win_strides = np.array(win_strides)
    a_shape = np.array(a.shape)
    a_strides = np.array(a.strides)
    bh_shape = np.concatenate((win_shape, np.divide(a_shape - win_shape, win_strides).astype(int) + 1))
    bh_strides = np.concatenate((a_strides, np.multiply(win_strides, a_strides)))
    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)

def hamming2d(a, b):
    w2d = np.outer(np.hamming(a), np.hamming(b))
    return np.sqrt(w2d)

def crop2d(data, center_r=15):
    nx, ny = data.shape[0:2]
    if center_r > 0:
        cx = int(nx/2)
        cy = int(ny/2)
        cxr = np.arange(round(cx-center_r), round(cx+center_r))
        cyr = np.arange(round(cy-center_r), round(cy+center_r))
    return data[np.ix_(cxr.astype(int), cyr.astype(int))]

def mask2d(nx, ny, center_r=15, undersampling=0.5):
    k = int(round(nx*ny*undersampling))
    ri = np.random.choice(nx*ny, k, replace=False)
    ma = np.zeros(nx*ny)
    ma[ri] = 1
    mask = ma.reshape((nx, ny))
    if center_r > 0:
        cx = int(nx/2)
        cy = int(ny/2)
        cxr = np.arange(round(cx-center_r), round(cx+center_r+1))
        cyr = np.arange(round(cy-center_r), round(cy+center_r+1))
        mask[np.ix_(cxr.astype(int), cyr.astype(int))] = np.ones((cxr.shape[0], cyr.shape[0]))
    return mask

def pad2d(data, nx, ny):
    datsize = data.shape
    padsize = np.array(datsize)
    padsize[0] = nx
    padsize[1] = ny
    ndata = np.zeros(tuple(padsize), dtype=data.dtype)
    datrx = int(datsize[0]/2)
    datry = int(datsize[1]/2)
    cx = int(nx/2)
    cy = int(ny/2)
    cxr = np.arange(round(cx-datrx), round(cx-datrx+datsize[0]))
    cyr = np.arange(round(cy-datry), round(cy-datry+datsize[1]))
    ndata[np.ix_(cxr.astype(int), cyr.astype(int))] = data
    return ndata

def optscaling(FT, b):
    x0 = np.absolute(FT.backward(b))
    return max(x0.flatten())

class FFT2d:
    def __init__(self, axes=(0, 1)):
        self.axes = axes
    def forward(self, im):
        im = np.fft.fftshift(im, self.axes)
        ksp = np.fft.fft2(im, s=None, axes=self.axes)
        ksp = np.fft.ifftshift(ksp, self.axes)
        return ksp
    def backward(self, ksp):
        ksp = np.fft.fftshift(ksp, self.axes)
        im = np.fft.ifft2(ksp, s=None, axes=self.axes)
        im = np.fft.ifftshift(im, self.axes)
        return im

class FFT2d_kmask:
    def __init__(self, mask, axes=(0, 1)):
        self.mask = mask
        self.axes = axes
    def forward(self, im):
        im = np.fft.fftshift(im, self.axes)
        ksp = np.fft.fft2(im, s=None, axes=self.axes)
        ksp = np.fft.ifftshift(ksp, self.axes)
        if len(ksp.shape) != len(self.mask.shape):
            ksp_out_shape, mask_out_shape = dim_match(ksp.shape, self.mask.shape)
            mksp = np.multiply(ksp.reshape(ksp_out_shape), self.mask.reshape(mask_out_shape))
        else:
            mksp = np.multiply(ksp, self.mask)
        return mksp
    def backward(self, ksp):
        ksp = np.fft.fftshift(ksp, self.axes)
        im = np.fft.ifft2(ksp, s=None, axes=self.axes)
        im = np.fft.ifftshift(im, self.axes)
        return im

def espirit_2d(xcrop, x_shape, nsingularv=150, hkwin_shape=(16, 16), pad_before_espirit=0, pad_fact=1, sigv_th=0.01, nsigv_th=0.2):
    ft = FFT2d()
    h = hankelnd_r(xcrop, (hkwin_shape[0], hkwin_shape[1], 1))
    dimh = h.shape
    hmtx = h.reshape((dimh[0]*dimh[1]*dimh[2], dimh[3], dimh[4], dimh[5])).reshape((dimh[0]*dimh[1]*dimh[2], dimh[3]*dimh[4]*dimh[5]))
    U, s, V = np.linalg.svd(hmtx, full_matrices=False)
    for k in range(len(s)):
        if s[k] > s[0]*nsigv_th:
            nsingularv = k
    vn = V[0:nsingularv, :].reshape((nsingularv, dimh[3], dimh[4], dimh[5])).transpose((1, 2, 0, 3))
    if pad_before_espirit == 0:
        nx = min(pad_fact * xcrop.shape[0], x_shape[0])
        ny = min(pad_fact * xcrop.shape[1], x_shape[1])
    else:
        nx = x_shape[0]
        ny = x_shape[1]
    nc = x_shape[2]
    hwin = hamming2d(vn.shape[0], vn.shape[1])
    vn = np.multiply(vn, hwin[:, :, np.newaxis, np.newaxis])
    vn = pad2d(vn, nx, ny)
    imvn = ft.backward(vn)
    sim = np.zeros((nx, ny), dtype=np.complex128)
    Vim = np.zeros((nx, ny, nc), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            vpix = imvn[ix, iy, :, :].squeeze()
            vpix = np.matrix(vpix).transpose()
            vvH = vpix.dot(vpix.getH())
            U, s, V = np.linalg.svd(vvH, full_matrices=False)
            sim[ix, iy] = s[0]
            Vim[ix, iy, :] = V[0, :].squeeze()
    Vim = np.conj(Vim)
    if pad_before_espirit == 0:
        Vim = ft.backward(pad2d(ft.forward(Vim), x_shape[0], x_shape[1]))
        sim = ft.backward(pad2d(ft.forward(sim), x_shape[0], x_shape[1]))
    Vimnorm = np.linalg.norm(Vim, axis=2)
    Vim = np.divide(Vim, 1e-6 + Vimnorm[:, :, np.newaxis])
    sim = sim/np.max(sim.flatten())
    for ix in range(x_shape[0]):
        for iy in range(x_shape[1]):
            if sim[ix, iy] < sigv_th:
                Vim[ix, iy, :] = np.zeros(nc)
    return Vim, np.absolute(sim)

def load_and_preprocess_data(data_path, undersampling_rate=0.25):
    """
    Loads .mat file, crops calibration data, estimates sensitivity maps,
    generates mask, and prepares observed data 'b' and ground truth.
    """
    print(f"Loading data from {data_path}")
    mat_contents = sio.loadmat(data_path)
    x = mat_contents["DATA"] # K-space data (multicoil)
    nx, ny, nc = x.shape
    
    # Crop k-space for ESPIRiT calibration
    print("Cropping data for calibration...")
    xcrop = crop2d(x, 16)
    
    print("Estimating coil sensitivities using ESPIRiT...")
    Vim, sim = espirit_2d(xcrop, x.shape, nsingularv=150, hkwin_shape=(16,16), 
                          pad_before_espirit=0, pad_fact=2)
    
    # Create mask
    print("Creating undersampling mask...")
    mask = mask2d(nx, ny, center_r=15, undersampling=undersampling_rate)
    
    # Create Ground Truth (Reference)
    # Using SoS of fully sampled data
    ft_op = FFT2d()
    im_full = ft_op.backward(x)
    im_ref = np.sqrt(np.sum(np.abs(im_full)**2, axis=2))
    
    # Generate undersampled measurements (Simulate acquisition)
    b_multicoil = np.multiply(x, mask[:,:,np.newaxis])
    
    # Scaling
    # We need a temporary operator to compute optimal scaling
    ft_temp = FFT2d_kmask(mask)
    scaling_factor = optscaling(ft_temp, b_multicoil)
    print(f"Scaling factor: {scaling_factor}")
    b_scaled = b_multicoil / scaling_factor
    
    return b_scaled, mask, Vim, im_ref
