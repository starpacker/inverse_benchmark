import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

from scipy.optimize import minimize as sp_minimize

def run_inversion(q_data, I_data, density_gt_shape, voxel_size, q_max, n_q, n_iter, n_runs, seed):
    """
    SAXS density reconstruction via gradient-based optimization.

    Minimizes:
        L(ρ) = Σ_i [ I_model(q_i) - I_data(q_i) ]²

    where I_model is the radially-averaged power spectrum of ρ,
    subject to positivity and compact support constraints.

    Parameters
    ----------
    q_data : ndarray
        q values from measured data.
    I_data : ndarray
        Measured I(q) values.
    density_gt_shape : tuple
        Expected output shape (N, N, N).
    voxel_size : float
        Voxel size in Angstroms.
    q_max : float
        Maximum q value.
    n_q : int
        Number of q bins.
    n_iter : int
        Maximum number of iterations.
    n_runs : int
        Number of independent optimization runs.
    seed : int
        Random seed.

    Returns
    -------
    ndarray
        Reconstructed 3D electron density.
    """
    print(f"[RECON] Gradient-based SAXS reconstruction ({n_iter} iterations) ...")
    N = density_gt_shape[0]

    # q-grid for radial binning
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Radial bin assignments
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    bin_index = np.full(q_3d.shape, -1, dtype=int)
    bin_count = np.zeros(n_q)
    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        bin_index[mask] = i
        bin_count[i] = mask.sum()

    # Target I(q) - normalize
    I_target = I_data.copy()
    I_target = I_target / max(I_target.max(), 1e-12)

    # Support mask (spherical)
    z, y, x = np.mgrid[:N, :N, :N] - N // 2
    r2 = x ** 2 + y ** 2 + z ** 2
    support = r2 < (N // 2 - 1) ** 2
    support_flat = support.ravel()
    n_support = int(support_flat.sum())

    def compute_Iq_and_grad(rho_in):
        """Compute I(q) from rho and gradient of L w.r.t. rho."""
        F = fftshift(fftn(ifftshift(rho_in)))
        I_3d = np.abs(F) ** 2

        # Radial average → I_model(q)
        I_model = np.zeros(n_q)
        for i in range(n_q):
            if bin_count[i] > 0:
                I_model[i] = np.mean(I_3d[bin_index == i])

        # Normalize model I(q)
        I_model_max = max(I_model.max(), 1e-12)
        I_model_norm = I_model / I_model_max

        # Residual per bin
        residual_per_bin = I_model_norm - I_target

        # Loss = sum of squared residuals + Tikhonov
        alpha_smooth = 1e-4
        loss = np.sum(residual_per_bin ** 2) + alpha_smooth * np.sum(rho_in ** 2)

        # Gradient: ∂L/∂ρ via adjoint
        dL_dI3d = np.zeros_like(I_3d)
        for i in range(n_q):
            if bin_count[i] > 0:
                mask_i = bin_index == i
                dL_dI3d[mask_i] = (2.0 / I_model_max) * residual_per_bin[i] / bin_count[i]

        # Chain rule through |F|^2 = F * conj(F)
        grad_F = dL_dI3d * np.conj(F)

        # Adjoint of FFT
        grad_rho = 2.0 * np.real(fftshift(ifftn(ifftshift(grad_F)))) * N ** 3
        # Tikhonov gradient
        grad_rho += 2 * alpha_smooth * rho_in

        return loss, I_model_norm, grad_rho

    def run_lbfgsb(seed_offset):
        """Single L-BFGS-B run with given seed offset."""
        rng_run = np.random.default_rng(seed + 200 + seed_offset)
        # Start with Gaussian blob + random perturbation
        rho_init = np.exp(-r2 / (2 * (N / 6) ** 2)).astype(np.float64)
        rho_init += 0.05 * rng_run.random((N, N, N))
        rho_init = np.maximum(rho_init, 0)
        rho_init = rho_init / max(rho_init.max(), 1e-12)

        x0 = rho_init.ravel()[support_flat].astype(np.float64)

        def obj(x_flat):
            rho_full = np.zeros(N ** 3, dtype=np.float64)
            rho_full[support_flat] = x_flat
            rho_3d = rho_full.reshape((N, N, N))
            loss, _, grad_3d = compute_Iq_and_grad(rho_3d)
            grad_flat = grad_3d.ravel()[support_flat].copy()
            return float(loss), np.ascontiguousarray(grad_flat, dtype=np.float64)

        result = sp_minimize(
            obj, x0, method='L-BFGS-B', jac=True,
            bounds=[(0, None)] * n_support,
            options={'maxiter': 1000, 'maxfun': 20000,
                     'ftol': 0, 'gtol': 1e-12, 'maxcor': 20}
        )
        return result

    print(f"[RECON] Running {n_runs} L-BFGS-B runs on {n_support} support voxels ...")

    best_result = None
    best_loss = float('inf')

    for run_idx in range(n_runs):
        result = run_lbfgsb(run_idx * 13)
        print(f"[RECON] Run {run_idx + 1}/{n_runs}: loss={result.fun:.6f}  "
              f"iters={result.nit}  fevals={result.nfev}")
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    print(f"[RECON] Best loss: {best_loss:.6f}")

    rho_full = np.zeros(N ** 3)
    rho_full[support_flat] = best_result.x
    rho = rho_full.reshape((N, N, N))

    # Normalize
    if rho.max() > 0:
        rho = rho / rho.max()

    print(f"[RECON] Final range: [{rho.min():.4f}, {rho.max():.4f}]")
    return rho
