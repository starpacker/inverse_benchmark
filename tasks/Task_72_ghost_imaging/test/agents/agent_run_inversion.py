import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import dct, idct

def run_inversion(data, method='fista', n_iter=500, lam=0.01, tv_weight=0.03, tv_iter=50):
    """
    Run ghost imaging inversion using specified method.
    
    Args:
        data: dict from load_and_preprocess_data
        method: 'correlation', 'ista', or 'fista'
        n_iter: number of iterations for iterative methods
        lam: regularization parameter
        tv_weight: TV denoising weight
        tv_iter: TV denoising iterations
    
    Returns:
        dict containing:
            - img_rec: reconstructed image (2D)
            - x_rec: vectorized reconstruction
            - method: method used
    """
    Phi = data['Phi']
    b = data['b_noisy']
    img_size = data['config']['img_size']
    M, N = Phi.shape
    
    def dct2d_transform(x, size, inverse=False):
        """2D DCT sparsifying transform."""
        X = x.reshape(size, size)
        if inverse:
            return idct(idct(X, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()
        else:
            return dct(dct(X, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()
    
    def soft_threshold(x, tau):
        """Soft thresholding (proximal operator for L1)."""
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
    
    def tv_denoise(img, weight, n_iter_tv):
        """Isotropic TV denoising (Chambolle's projection)."""
        u = img.copy()
        px = np.zeros_like(u)
        py = np.zeros_like(u)
        tau = 0.25
        
        for _ in range(n_iter_tv):
            gx = np.diff(u, axis=0, append=u[-1:, :])
            gy = np.diff(u, axis=1, append=u[:, -1:])
            
            px_new = px + tau * gx
            py_new = py + tau * gy
            norm = np.sqrt(px_new**2 + py_new**2)
            norm = np.maximum(norm / weight, 1)
            px = px_new / norm
            py = py_new / norm
            
            div_x = np.diff(px, axis=0, prepend=np.zeros((1, u.shape[1])))
            div_y = np.diff(py, axis=1, prepend=np.zeros((u.shape[0], 1)))
            u = img + weight * (div_x + div_y)
        
        return u
    
    if method == 'correlation':
        # Traditional correlation ghost imaging
        b_mean = b.mean()
        Phi_mean = Phi.mean(axis=0)
        x_rec = np.zeros(N)
        for i in range(M):
            x_rec += (b[i] - b_mean) * (Phi[i] - Phi_mean)
        x_rec /= M
        x_rec = np.maximum(x_rec, 0)
        img_rec = x_rec.reshape(img_size, img_size)
        
    elif method == 'ista':
        # ISTA for compressed sensing
        L = np.linalg.norm(Phi.T @ Phi, ord=2)
        step = 1.0 / L
        x = np.zeros(N)
        
        print(f"  ISTA: {n_iter} iterations, λ={lam:.4f}, step={step:.6f}")
        
        for it in range(n_iter):
            residual = Phi @ x - b
            grad = Phi.T @ residual
            z = x - step * grad
            
            z_dct = dct2d_transform(z, img_size)
            z_dct = soft_threshold(z_dct, lam * step)
            x = dct2d_transform(z_dct, img_size, inverse=True)
            x = np.maximum(x, 0)
            
            if (it + 1) % 50 == 0:
                obj = 0.5 * np.linalg.norm(residual)**2 + lam * np.sum(np.abs(z_dct))
                print(f"    iter {it+1:4d}: obj={obj:.4f}")
        
        x_rec = x
        img_rec = x_rec.reshape(img_size, img_size)
        img_rec = np.clip(img_rec, 0, 1)
        img_rec = tv_denoise(img_rec, tv_weight, tv_iter)
        x_rec = img_rec.ravel()
        
    elif method == 'fista':
        # FISTA with Nesterov acceleration
        L = np.linalg.norm(Phi.T @ Phi, ord=2)
        step = 1.0 / L
        x = np.zeros(N)
        y = x.copy()
        t = 1.0
        
        print(f"  FISTA: {n_iter} iterations, λ={lam:.4f}")
        
        for it in range(n_iter):
            residual = Phi @ y - b
            grad = Phi.T @ residual
            z = y - step * grad
            
            z_dct = dct2d_transform(z, img_size)
            z_dct = soft_threshold(z_dct, lam * step)
            x_new = dct2d_transform(z_dct, img_size, inverse=True)
            x_new = np.maximum(x_new, 0)
            
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            
            x = x_new
            t = t_new
            
            if (it + 1) % 100 == 0:
                obj = 0.5 * np.linalg.norm(Phi @ x - b)**2 + lam * np.sum(np.abs(z_dct))
                print(f"    iter {it+1:4d}: obj={obj:.4f}")
        
        x_rec = x
        img_rec = x_rec.reshape(img_size, img_size)
        img_rec = np.clip(img_rec, 0, 1)
        img_rec = tv_denoise(img_rec, tv_weight, tv_iter)
        x_rec = img_rec.ravel()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'img_rec': img_rec,
        'x_rec': x_rec,
        'method': method
    }
