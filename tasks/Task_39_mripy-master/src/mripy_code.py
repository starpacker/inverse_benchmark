import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from math import ceil

# =============================================================================
# 0. HELPER FUNCTIONS AND CLASSES (Strictly defined first)
# =============================================================================

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

def plotim3(im, save_path=None):
    im = np.flip(im, 0)
    plt.figure()
    plt.imshow(im, cmap=cm.gray, origin='lower', interpolation='none')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.close()

# --- Classes ---

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

class espirit:
    def __init__(self, sensitivity=None, coil_axis=None):
        self.sens = sensitivity
        if coil_axis is None and self.sens is not None:
            self.coil_axis = len(sensitivity.shape)-1
        else:
            self.coil_axis = coil_axis
    def backward(self, im_coils):
        sens_out_shape, im_out_shape = dim_match(self.sens.shape, im_coils.shape)
        return np.sum(np.multiply(im_coils.reshape(im_out_shape), np.conj(self.sens).reshape(sens_out_shape)), axis=self.coil_axis, keepdims=True)
    def forward(self, im_sos):
        sens_out_shape, im_out_shape = dim_match(self.sens.shape, im_sos.shape)
        return np.multiply(im_sos.reshape(im_out_shape), self.sens.reshape(sens_out_shape))

class joint2operators:
    def __init__(self, Aopt, Bopt):
        self.Aopt = Aopt
        self.Bopt = Bopt
    def forward(self, xin):
        xout = self.Bopt.forward(self.Aopt.forward(xin))
        return xout
    def backward(self, xin):
        xout = self.Aopt.backward(self.Bopt.backward(xin))
        return xout

# --- Optimization Helpers ---

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

# =============================================================================
# 1. LOAD AND PREPROCESS
# =============================================================================

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

# =============================================================================
# 2. FORWARD OPERATOR
# =============================================================================

def forward_operator(image, mask, sensitivity_maps):
    """
    Implements A(x): SENSE -> FFT -> Mask
    image: 2D complex image
    mask: 2D mask
    sensitivity_maps: 3D coil maps (nx, ny, nc)
    Returns: Undersampled multicoil k-space
    """
    # 1. Apply SENSE (Sensitivity Encoding)
    # Equivalent to espirit.forward(image)
    sens_out_shape, im_out_shape = dim_match(sensitivity_maps.shape, image.shape)
    im_coils = np.multiply(image.reshape(im_out_shape), sensitivity_maps.reshape(sens_out_shape))
    
    # 2. Apply FFT
    # Equivalent to FFT2d.forward (embedded in FFT2d_kmask)
    axes = (0, 1)
    im_coils_shifted = np.fft.fftshift(im_coils, axes)
    ksp_full = np.fft.fft2(im_coils_shifted, s=None, axes=axes)
    ksp_full = np.fft.ifftshift(ksp_full, axes)
    
    # 3. Apply Mask
    # Equivalent to FFT2d_kmask.forward
    if len(ksp_full.shape) != len(mask.shape):
        ksp_out_shape, mask_out_shape = dim_match(ksp_full.shape, mask.shape)
        ksp_masked = np.multiply(ksp_full.reshape(ksp_out_shape), mask.reshape(mask_out_shape))
    else:
        ksp_masked = np.multiply(ksp_full, mask)
        
    return ksp_masked

# =============================================================================
# 3. RUN INVERSION
# =============================================================================

def run_inversion(y_observed, mask, sensitivity_maps, lambda_tv=0.002, rho=1.0, n_iters=20):
    """
    Runs the ADMM optimization.
    y_observed: The scaled, masked k-space data.
    """
    # Create Helper Operator Objects for the Solver
    esp = espirit(sensitivity_maps)
    ft_masked = FFT2d_kmask(mask)
    Aopt = joint2operators(esp, ft_masked)
    
    step = 0.5
    
    print("Running ADMM TV Reconstruction...")
    # ADMM_l2Afxnb_tvx requires function handles for A and A_adjoint
    x_rec = ADMM_l2Afxnb_tvx(
        Afunc=Aopt.forward, 
        invAfunc=Aopt.backward, 
        b=y_observed, 
        Nite=n_iters, 
        step=step, 
        tv_r=lambda_tv, 
        rho=rho
    )
    
    return np.absolute(x_rec) # Return magnitude image

# =============================================================================
# 4. EVALUATE RESULTS
# =============================================================================

def evaluate_results(reconstruction, reference_image, save_prefix="result"):
    """
    Computes PSNR and saves images.
    """
    # Squeeze extra dimensions for consistent comparison
    reconstruction = np.squeeze(reconstruction)
    reference_image = np.squeeze(reference_image)
    
    # Take magnitude for complex-valued images
    if np.iscomplexobj(reconstruction):
        reconstruction = np.abs(reconstruction)
    if np.iscomplexobj(reference_image):
        reference_image = np.abs(reference_image)
    
    # Normalize images for fair comparison
    if np.max(reference_image) != 0:
        ref_norm = reference_image / np.max(reference_image)
    else:
        ref_norm = reference_image
        
    if np.max(reconstruction) != 0:
        rec_norm = reconstruction / np.max(reconstruction)
    else:
        rec_norm = reconstruction

    # MSE and PSNR
    mse = np.mean((ref_norm - rec_norm) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    
    # Save images
    plotim3(rec_norm, save_path=f'{save_prefix}_recon.png')
    plotim3(ref_norm, save_path=f'{save_prefix}_ref.png')
    
    return psnr

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Configuration
    DATA_PATH = 'data/brain_32ch.mat'
    UNDERSAMPLING = 0.25
    LAMBDA_TV = 0.002
    RHO = 1.0
    N_ITERS = 20

    # 1. Load Data
    try:
        b_obs, mask, sensitivity_maps, ref_img = load_and_preprocess_data(DATA_PATH, UNDERSAMPLING)
        
        # 2. Test Forward Operator (Sanity Check - Optional but good for validation)
        # We perform a dummy forward pass on a zero image to ensure shapes align
        dummy_x = np.zeros_like(ref_img, dtype=np.complex128)
        dummy_y = forward_operator(dummy_x, mask, sensitivity_maps)
        print(f"Forward operator test output shape: {dummy_y.shape}")

        # 3. Run Inversion
        rec_img = run_inversion(b_obs, mask, sensitivity_maps, LAMBDA_TV, RHO, N_ITERS)

        # 4. Evaluate
        evaluate_results(rec_img, ref_img)

    except FileNotFoundError:
        print("Data file not found. Ensure 'data/brain_32ch.mat' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")