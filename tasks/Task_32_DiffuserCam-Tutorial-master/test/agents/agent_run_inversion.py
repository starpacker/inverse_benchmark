import numpy as np

import numpy.fft as fft

def SoftThresh(x, tau):
    return np.sign(x) * np.maximum(0, np.abs(x) - tau)

def Psi(v):
    # Forward finite difference
    return np.stack((np.roll(v, 1, axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)

def PsiT(U):
    # Adjoint of forward finite difference
    diff1 = np.roll(U[..., 0], -1, axis=0) - U[..., 0]
    diff2 = np.roll(U[..., 1], -1, axis=1) - U[..., 1]
    return diff1 + diff2

def C(M_arr, full_size, sensor_size):
    # Crop operator
    top = (full_size[0] - sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    return M_arr[top:top+sensor_size[0], left:left+sensor_size[1]]

def CT(b, full_size, sensor_size):
    # Transpose of Crop (Zero Pad)
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    # Create full zero array and place b in center
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top+sensor_size[0], pad_left:pad_left+sensor_size[1]] = b
    return out

def M_func(vk, H_fft):
    # Convolution operator in freq domain
    # M(v) = Real(IFFT( FFT(v) * H ))
    # Note: Logic must match original: ifftshift before fft, fftshift after ifft
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk)) * H_fft)))

def MT_func(x, H_fft):
    # Adjoint of convolution
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def precompute_H_fft(psf, full_size, sensor_size):
    # H = FFT( ZeroPad(PSF) )
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))

def run_inversion(data, psf, config):
    """
    Performs ADMM deconvolution.
    """
    # Dimensions
    sensor_size = np.array(psf.shape)
    full_size = 2 * sensor_size
    
    # Parameters
    mu1 = config['mu1']
    mu2 = config['mu2']
    mu3 = config['mu3']
    tau = config['tau']
    iters = config['iters']

    # Precomputations
    H_fft = precompute_H_fft(psf, full_size, sensor_size)
    
    # X_divmat
    ones_padded = CT(np.ones(sensor_size), full_size, sensor_size)
    X_divmat = 1.0 / (ones_padded + mu1)
    
    # PsiTPsi and R_divmat
    PsiTPsi_spatial = np.zeros(full_size)
    PsiTPsi_spatial[0,0] = 4
    PsiTPsi_spatial[0,1] = PsiTPsi_spatial[1,0] = PsiTPsi_spatial[0,-1] = PsiTPsi_spatial[-1,0] = -1
    PsiTPsi = fft.fft2(PsiTPsi_spatial)
    
    MTM_component = mu1 * (np.abs(np.conj(H_fft) * H_fft))
    PsiTPsi_component = mu2 * np.abs(PsiTPsi)
    id_component = mu3
    R_divmat = 1.0 / (MTM_component + PsiTPsi_component + id_component)

    # Initialization
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)
    xi = np.zeros_like(M_func(V, H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)

    print(f"Starting ADMM for {iters} iterations...")
    
    # Optimization Loop
    for i in range(iters):
        # 1. U update
        # u = SoftThresh(Psi(v) + eta/mu2, tau/mu2)
        U = SoftThresh(Psi(V) + eta/mu2, tau/mu2)
        
        # 2. X update
        # x = X_divmat * (xi + mu1*M(v) + CT(b))
        term_M = M_func(V, H_fft)
        term_CT = CT(data, full_size, sensor_size)
        X = X_divmat * (xi + mu1 * term_M + term_CT)
        
        # 3. V update
        # r = (mu3*w - rho) + PsiT(mu2*u - eta) + MT(mu1*x - xi)
        # v = IFFT( R_divmat * FFT(r) )
        term_w_rho = (mu3 * W - rho)
        term_psiT = PsiT(mu2 * U - eta)
        term_MT = MT_func(mu1 * X - xi, H_fft)
        r = term_w_rho + term_psiT + term_MT
        
        freq_space_result = R_divmat * fft.fft2(fft.ifftshift(r))
        V = np.real(fft.fftshift(fft.ifft2(freq_space_result)))
        
        # 4. W update
        # w = max(rho/mu3 + v, 0)
        W = np.maximum(rho/mu3 + V, 0)
        
        # 5. Multiplier updates
        xi = xi + mu1 * (M_func(V, H_fft) - X)
        eta = eta + mu2 * (Psi(V) - U)
        rho = rho + mu3 * (V - W)
        
        if i % 5 == 0:
            print(f"Iter {i}/{iters}")

    # Final result extraction
    recon = C(V, full_size, sensor_size)
    recon[recon < 0] = 0
    return recon
