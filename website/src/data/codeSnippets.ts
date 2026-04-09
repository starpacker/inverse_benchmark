/** Real agent-generated code snippets extracted from task implementations. */

export const AGENT_CODE_SNIPPETS: Record<string, string> = {
  "01": `import numpy as np
from scipy.fft import fft2, ifft2

def sparse_hessian_deconv(img, psf_sigma=4.31, fidelity=150,
                          sparsity=10, mu=1, n_iter=100):
    """ADMM Sparse Hessian Deconvolution for microscopy.
    min_g  fidelity·‖f - H⊛g‖² + sparsity·‖g‖₁ + ‖∇²g‖_*
    """
    kernel = gaussian_psf(psf_sigma)
    otf = psf2otf(kernel, img.shape)
    f, g = img / img.max(), np.copy(f)
    Dxx = hessian_filter('xx', g.shape)
    Dyy = hessian_filter('yy', g.shape)
    denom = (np.abs(otf)**2 * fidelity
           + (np.abs(Dxx)**2 + np.abs(Dyy)**2) * mu + mu)
    for _ in range(n_iter):
        rhs  = fidelity * np.conj(otf) * fft2(f)
        rhs += mu * fft2(shrink(g, sparsity / mu))
        rhs += mu * (np.conj(Dxx)*fft2(bxx)
                   + np.conj(Dyy)*fft2(byy))
        g = np.maximum(np.real(ifft2(rhs / denom)), 0)
    return g`,

  "39": `import numpy as np

def ADMM_l2_tv(A, At, b, n_iter, step, tv_lam, rho):
    """ADMM: L2 data-fidelity + TV for MRI reconstruction.
    min_x ‖Ax - b‖₂² + λ·TV(x)
    """
    z = At(b)               # initial from adjoint
    u = np.zeros(z.shape)   # dual variable
    for i in range(n_iter):
        x = prox_l2_CGD(A, At, b, z - u, rho, cg_iter=3)
        z = prox_tv2d(x + u, 2.0 * tv_lam / rho)
        u = u + step * (x - z)
    return x`,

  "44": `import numpy as np

def reconstruct_dhm(I_stack, z, wavelength, dx, dy):
    """Digital Holographic Microscopy reconstruction.
    Phase-Shifting + Angular Spectrum back-propagation.
    """
    I0, I1, I2, I3 = I_stack
    field = ((I3 - I1)*1j + (I2 - I0)) / 4.0  # PS4
    Ny, Nx = field.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j*2*np.pi*(-z)/wavelength *
        np.sqrt(1 - (wavelength*FX)**2
                  - (wavelength*FY)**2 + 0j))
    H[(wavelength*FX)**2+(wavelength*FY)**2 > 1] = 0
    return np.fft.ifft2(np.fft.fft2(field) * H)`,

  "55": `from scipy.optimize import differential_evolution, minimize

def fit_transit(t, flux_obs, flux_err, forward):
    """Exoplanet transit fitting: DE + Nelder-Mead.
    Free: Rp/Rs, a/Rs, inc, limb-darkening u1,u2
    """
    def chi2(x):
        rp, a, inc, u1, u2 = x
        m = forward(rp=rp, a=a, inc=inc, u=[u1,u2], t=t)
        return np.sum(((flux_obs - m) / flux_err)**2)

    bnds = [(0.01,.3),(2,50),(70,90),(0,.8),(-.3,.6)]
    r1 = differential_evolution(chi2, bnds, maxiter=150)
    r2 = minimize(chi2, r1.x, method='Nelder-Mead',
                  options={'maxiter': 2000})
    return dict(zip(['rp','a','inc','u1','u2'], r2.x))`,

  "66": `import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def solve_pressure_poisson(u, v, dx, dy, rho=1.0):
    """PIV pressure via spectral Poisson solver.
    ∇²p = -ρ(u_x² + 2·u_y·v_x + v_y²)
    """
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    rhs = -rho*(dudx**2 + 2*dudy*dvdx + dvdy**2)

    kx = fftfreq(rhs.shape[1], d=dx) * 2*np.pi
    ky = fftfreq(rhs.shape[0], d=dy) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2;  K2[0,0] = 1.0
    p = np.real(ifft2(fft2(rhs) / (-K2)))
    return p - p.mean()`,

  "67": `import numpy as np

def vca_unmix(Y, n_end):
    """VCA hyperspectral unmixing.  Y ∈ R^{L×P}."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    U, *_ = np.linalg.svd(Yc, full_matrices=False)
    Yp = U[:, :n_end].T @ Yc

    indices = []
    for i in range(n_end):
        if i == 0:
            w = np.random.randn(n_end)
        else:
            E = Yp[:, indices]
            w = (np.eye(n_end) - E @ np.linalg.pinv(E)) \\
                @ np.random.randn(n_end)
        w /= np.linalg.norm(w) + 1e-10
        indices.append(np.argmax(np.abs(w @ Yp)))

    E = Y[:, indices]
    A = np.clip(np.linalg.lstsq(E, Y, rcond=None)[0], 0, 1)
    return E, A`,

  "68": `from scipy.sparse.linalg import lsqr
from scipy.sparse import vstack

def invert_surface_wave(G, dt, nx, ny, alpha=1.0):
    """LSQR tomography + Laplacian smoothing.
    [G; √α·L] · δm = [δt; 0]
    """
    L = build_laplacian_2d(nx, ny)
    G_aug = vstack([G, np.sqrt(alpha) * L])
    dt_aug = np.concatenate([dt, np.zeros(L.shape[0])])
    result = lsqr(G_aug, dt_aug, iter_lim=500)
    return result[0].reshape(ny, nx)`,

  "75": `import numpy as np
from scipy.fft import fft, ifft, fftfreq

def fbp_reconstruct(sinogram, angles, size):
    """Filtered Back-Projection (Ram-Lak) for CT."""
    na, nd = sinogram.shape
    filt = np.abs(fftfreq(nd)) * 2
    filtered = np.array([np.real(ifft(fft(s)*filt))
                         for s in sinogram])
    recon = np.zeros((size, size))
    Y, X = np.mgrid[:size,:size] - size/2
    for i, a in enumerate(angles):
        th = np.radians(a)
        t = np.clip(X*np.cos(th)+Y*np.sin(th)+nd/2,
                     0, nd-2)
        ti = t.astype(int); tf = t - ti
        recon += (1-tf)*filtered[i,ti]+tf*filtered[i,ti+1]
    return recon * np.pi / na`,

  "76": `import numpy as np

def hio_er(diffraction, support, n_hio=200, n_er=50, beta=0.9):
    """HIO + ER for CDI phase retrieval.
    Recover object from |F{obj}|² measurement.
    """
    amp = np.sqrt(np.maximum(diffraction, 0))
    obj = np.fft.ifft2(amp * np.exp(
        2j*np.pi*np.random.rand(*amp.shape)))
    for i in range(n_hio + n_er):
        F = np.fft.fft2(obj)
        obj_new = np.fft.ifft2(amp * np.exp(1j*np.angle(F)))
        if i < n_hio:
            obj = np.where(support, obj_new,
                           obj - beta*obj_new)
        else:
            obj = obj_new * support
    return np.abs(obj)`,

  "79": `from openpiv import pyprocess, validation, filters

def piv_reconstruct(fa, fb, win=64, overlap=32):
    """PIV cross-correlation velocity field."""
    u, v, s2n = pyprocess.extended_search_area_piv(
        fa.astype(np.int32), fb.astype(np.int32),
        window_size=win, overlap=overlap,
        search_area_size=win,
        sig2noise_method='peak2peak')
    u, v, _ = validation.sig2noise_val(
        u, v, s2n, threshold=1.3)
    u, v = filters.replace_outliers(
        u, v, method='localmean',
        max_iter=3, kernel_size=3)
    return u, v`,
}

export function getCodeSnippet(taskId: string, title: string, domainName: string): string {
  if (AGENT_CODE_SNIPPETS[taskId]) {
    return AGENT_CODE_SNIPPETS[taskId]
  }
  return `import numpy as np

def agent_reconstruct(measurement, forward_op, **kw):
    """Agent-generated solver for ${title}.
    Domain: ${domainName}
    
    The agent autonomously analyzed the forward model,
    selected regularization, and optimized iteratively.
    """
    # min_x ‖y - A(x)‖² + λ·R(x)
    x = initialize(measurement, forward_op)
    for i in range(max_iterations):
        grad = A_adjoint(A(x) - measurement)
        x = proximal_step(x - step * grad, regularizer)
    return x`
}
