import torch

import matplotlib.pyplot as plt

from math import pi

from typing import Optional, Union, Tuple, Literal, Annotated

from functools import lru_cache

from scipy.special import roots_legendre

from scipy.fft import next_fast_len

from torch import Tensor

from caskade import Module, forward, Param

c_Mpc_s = 9.71561189e-15

km_to_Mpc = 3.2407792896664e-20

def meshgrid(pixelscale, nx, ny=None, device=None, dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    if ny is None:
        ny = nx
    xs = torch.linspace(-1, 1, nx, device=device, dtype=dtype) * pixelscale * (nx - 1) / 2
    ys = torch.linspace(-1, 1, ny, device=device, dtype=dtype) * pixelscale * (ny - 1) / 2
    return torch.meshgrid([xs, ys], indexing="xy")

def _quad_table(n, p, dtype, device):
    abscissa, weights = roots_legendre(n)
    w = torch.tensor(weights, dtype=dtype, device=device)
    a = p * torch.tensor(abscissa, dtype=dtype, device=device) / 2.0
    X, Y = torch.meshgrid(a, a, indexing="xy")
    W = torch.outer(w, w) / 4.0
    X, Y = X.reshape(-1), Y.reshape(-1)
    return X, Y, W.reshape(-1)

def gaussian_quadrature_grid(pixelscale, X, Y, quad_level=3):
    abscissaX, abscissaY, weight = _quad_table(quad_level, pixelscale, dtype=X.dtype, device=X.device)
    Xs = torch.repeat_interleave(X[..., None], quad_level**2, -1) + abscissaX
    Ys = torch.repeat_interleave(Y[..., None], quad_level**2, -1) + abscissaY
    return Xs, Ys, weight

def gaussian_quadrature_integrator(F: Tensor, weight: Tensor):
    return (F * weight).sum(axis=-1)

def translate_rotate(x, y, x0, y0, phi: Optional[Tensor] = None):
    xt = x - x0
    yt = y - y0
    if phi is not None:
        c_phi = phi.cos()
        s_phi = phi.sin()
        return xt * c_phi + yt * s_phi, yt * c_phi - xt * s_phi
    return xt, yt

def derotate(vx, vy, phi: Optional[Tensor] = None):
    if phi is None:
        return vx, vy
    c_phi = phi.cos()
    s_phi = phi.sin()
    return vx * c_phi - vy * s_phi, vx * s_phi + vy * c_phi

def interp2d(im: Tensor, x: Tensor, y: Tensor, method: Literal["linear", "nearest"] = "linear", padding_mode: str = "zeros") -> Tensor:
    if im.ndim != 2:
        raise ValueError(f"im must be 2D (received {im.ndim}D tensor)")
    if padding_mode == "clamp":
        x = x.clamp(-1, 1)
        y = y.clamp(-1, 1)
    else:
        idxs_out_of_bounds = (y < -1) | (y > 1) | (x < -1) | (x > 1)

    h, w = im.shape
    x = 0.5 * ((x + 1) * w - 1)
    y = 0.5 * ((y + 1) * h - 1)

    if method == "nearest":
        result = im[y.round().long().clamp(0, h - 1), x.round().long().clamp(0, w - 1)]
    elif method == "linear":
        x0 = x.floor().long().clamp(0, w - 2)
        y0 = y.floor().long().clamp(0, h - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        fa = im[y0, x0]
        fb = im[y1, x0]
        fc = im[y0, x1]
        fd = im[y1, x1]
        dx1 = x1 - x
        dx0 = x - x0
        dy1 = y1 - y
        dy0 = y - y0
        result = fa * dx1 * dy1 + fb * dx1 * dy0 + fc * dx0 * dy1 + fd * dx0 * dy0
    
    if padding_mode == "zeros":
        result = torch.where(idxs_out_of_bounds, torch.zeros_like(result), result)
    return result

def _h_poly(t):
    tt = t[None, :] ** (torch.arange(4, device=t.device)[:, None])
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    return A @ tt

def interp1d(x: Tensor, y: Tensor, xs: Tensor, extend: Literal["extrapolate", "const", "linear"] = "extrapolate") -> Tensor:
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    if extend == "const":
        ret[xs > x[-1]] = y[-1]
    elif extend == "linear":
        indices = xs > x[-1]
        ret[indices] = y[-1] + (xs[indices] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return ret

NameType = Annotated[Optional[str], "Name of the cosmology"]

class Cosmology(Module):
    def __init__(self, name: NameType = None):
        super().__init__(name)

    @forward
    def angular_diameter_distance(self, z: torch.Tensor) -> torch.Tensor:
        return self.comoving_distance(z) / (1 + z)

    @forward
    def angular_diameter_distance_z1z2(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.comoving_distance_z1z2(z1, z2) / (1 + z2)
    
    @forward
    def comoving_distance_z1z2(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.comoving_distance(z2) - self.comoving_distance(z1)

class FlatLambdaCDM(Cosmology):
    def __init__(self, h0=0.6766, critical_density_0=None, Om0=0.30966, name=None):
        super().__init__(name)
        self.h0 = Param("h0", torch.tensor(h0), units="unitless")
        self.Om0 = Param("Om0", torch.tensor(Om0), units="unitless")
        
        _x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
        from scipy.special import hyp2f1
        _y_grid = torch.as_tensor(_x_grid * hyp2f1(1/3, 1/2, 4/3, -(_x_grid**3)), dtype=torch.float64)
        self._comoving_distance_helper_x_grid = _x_grid.to(dtype=torch.float32)
        self._comoving_distance_helper_y_grid = _y_grid.to(dtype=torch.float32)

    def to(self, device=None, dtype=None):
        super().to(device, dtype)
        self._comoving_distance_helper_x_grid = self._comoving_distance_helper_x_grid.to(device, dtype)
        self._comoving_distance_helper_y_grid = self._comoving_distance_helper_y_grid.to(device, dtype)
        return self

    def hubble_distance(self, h0):
        return c_Mpc_s / (100 * km_to_Mpc) / h0

    @forward
    def _comoving_distance_helper(self, x: Tensor) -> Tensor:
        return interp1d(self._comoving_distance_helper_x_grid, self._comoving_distance_helper_y_grid, torch.atleast_1d(x)).reshape(x.shape)

    @forward
    def comoving_distance(self, z: Tensor, h0, Om0) -> Tensor:
        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        DH = self.hubble_distance(h0)
        DC1z = self._comoving_distance_helper((1 + z) * ratio)
        DC = self._comoving_distance_helper(ratio)
        return DH * (DC1z - DC) / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))

class Lens(Module):
    def __init__(self, cosmology, z_l=None, name=None, z_s=None):
        super().__init__(name)
        self.cosmology = cosmology
        self.z_l = Param("z_l", z_l, units="unitless", valid=(0, None))
        self.z_s = Param("z_s", z_s, units="unitless", valid=(0, None))

class ThinLens(Lens):
    pass

def reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, s=0.0):
    q = torch.where(q == 1.0, q - 1e-6, q)
    x, y = translate_rotate(x, y, x0, y0, phi)
    q2_ = q**2
    f = (1 - q2_).sqrt()
    rein_q_sqrt_f_ = Rein * q.sqrt() / f
    psi = (q2_ * (x**2 + s**2) + y**2).sqrt()
    ax = rein_q_sqrt_f_ * (f * x / (psi + s)).atan()
    ay = rein_q_sqrt_f_ * (f * y / (psi + q2_ * s)).atanh()
    return derotate(ax, ay, phi)

class SIE(ThinLens):
    def __init__(self, cosmology, z_l=None, z_s=None, x0=None, y0=None, q=None, phi=None, Rein=None, parametrization="Rein", sigma_v=None, angle_system="q_phi", s=0.0, name=None, **kwargs):
        super().__init__(cosmology, z_l, name=name, z_s=z_s)
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.q = Param("q", q, units="unitless", valid=(0, 1))
        self.phi = Param("phi", phi, units="radians", valid=(0, pi), cyclic=True)
        self.Rein = Param("Rein", Rein, units="arcsec", valid=(0, None))
        self.s = s

    @forward
    def reduced_deflection_angle(self, x, y, x0, y0, q, phi, Rein):
        return reduced_deflection_angle_sie(x0, y0, q, phi, Rein, x, y, self.s)

    @forward
    def raytrace(self, x, y, **kwargs):
        ax, ay = self.reduced_deflection_angle(x, y, **kwargs)
        return x - ax, y - ay

class Source(Module):
    pass

class Pixelated(Source):
    def __init__(self, image=None, x0=None, y0=None, pixelscale=None, scale=1.0, shape=None, name=None):
        if image is not None and image.ndim not in [2, 3]:
            raise ValueError(f"image must be 2D or 3D. Received {image.ndim}D tensor")
        super().__init__(name=name)
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self.image = Param("image", image, shape, units="flux")
        self.pixelscale = Param("pixelscale", pixelscale, units="arcsec/pixel", valid=(0, None))
        self.scale = Param("scale", scale, units="flux", valid=(0, None))

    @forward
    def brightness(self, x, y, x0, y0, image, pixelscale, scale, padding_mode="zeros"):
        fov_x = pixelscale * image.shape[1]
        fov_y = pixelscale * image.shape[0]
        return interp2d(image * scale, (x - x0).view(-1) / fov_x * 2, (y - y0).view(-1) / fov_y * 2, padding_mode=padding_mode).reshape(x.shape)

class LensSource(Module):
    def __init__(self, lens, source, pixelscale, pixels_x, lens_light=None, pixels_y=None, upsample_factor=1, quad_level=None, psf_mode="fft", psf_shape=None, psf=[[1.0]], x0=0.0, y0=0.0, name="sim"):
        super().__init__(name)
        self._psf_mode = psf_mode
        if psf is not None:
            psf = torch.as_tensor(psf)
        self._psf_shape = psf.shape if psf is not None else psf_shape
        self.psf = Param("psf", psf, self._psf_shape, units="unitless")
        self.x0 = Param("x0", x0, units="arcsec")
        self.y0 = Param("y0", y0, units="arcsec")
        self._pixelscale = pixelscale
        self.lens = lens
        self.source = source
        self.lens_light = lens_light
        self._pixels_x = pixels_x
        self._pixels_y = pixels_x if pixels_y is None else pixels_y
        self._upsample_factor = upsample_factor
        self._quad_level = quad_level
        self._build_grid()

    def to(self, device=None, dtype=None):
        super().to(device, dtype)
        self._grid = tuple(x.to(device, dtype) for x in self._grid)
        self._weights = self._weights.to(device, dtype)
        return self

    def _build_grid(self):
        self._psf_pad = (self._psf_shape[1] // 2, self._psf_shape[0] // 2)
        self._n_pix = (self._pixels_x * self._upsample_factor + self._psf_pad[0] * 2, self._pixels_y * self._upsample_factor + self._psf_pad[1] * 2)
        self._grid = meshgrid(self._pixelscale / self._upsample_factor, self._n_pix[0], self._n_pix[1])
        self._weights = torch.ones((1, 1), dtype=self._grid[0].dtype, device=self._grid[0].device)
        if self._quad_level is not None and self._quad_level > 1:
            finegrid_x, finegrid_y, weights = gaussian_quadrature_grid(self._pixelscale / self._upsample_factor, *self._grid, self._quad_level)
            self._grid = (finegrid_x, finegrid_y)
            self._weights = weights
        else:
            self._grid = (self._grid[0].unsqueeze(-1), self._grid[1].unsqueeze(-1))
        self._s = (next_fast_len(self._n_pix[0]), next_fast_len(self._n_pix[1]))

    def _fft2_padded(self, x):
        return torch.fft.rfft2(x, self._s)

    def _unpad_fft(self, x):
        return torch.roll(x, (-self._psf_pad[0], -self._psf_pad[1]), dims=(-2, -1))[..., : self._s[0], : self._s[1]]

    @forward
    def __call__(self, psf, x0, y0, source_light=True, lens_light=True, lens_source=True, psf_convolve=True, chunk_size=None):
        if self.source is None: source_light = False
        if self.lens_light is None: lens_light = False
        if psf.shape == (1, 1): psf_convolve = False

        grid = (self._grid[0] + x0, self._grid[1] + y0)

        if source_light:
            if lens_source:
                bx, by = self.lens.raytrace(grid[0].flatten(), grid[1].flatten())
                mu_fine = self.source.brightness(bx, by).reshape(grid[0].shape)
                mu = gaussian_quadrature_integrator(mu_fine, self._weights)
            else:
                mu_fine = self.source.brightness(grid[0].flatten(), grid[1].flatten()).reshape(grid[0].shape)
                mu = gaussian_quadrature_integrator(mu_fine, self._weights)
        else:
            mu = torch.zeros_like(grid[0][..., 0])

        if psf_convolve and self._psf_mode == "fft":
            mu_fft = self._fft2_padded(mu)
            psf_fft = self._fft2_padded(psf / psf.sum())
            mu = self._unpad_fft(torch.fft.irfft2(mu_fft * psf_fft, self._s).real)
        mu_clipped = mu[self._psf_pad[1] : self._pixels_y * self._upsample_factor + self._psf_pad[1], self._psf_pad[0] : self._pixels_x * self._upsample_factor + self._psf_pad[0]]
        mu_native_resolution = torch.nn.functional.avg_pool2d(mu_clipped[None, None], self._upsample_factor).squeeze(0).squeeze(0)
        return mu_native_resolution

def forward_operator(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Applies the lensing physics (forward model) to a candidate source image x.
    Returns the predicted lensed image y_pred.
    """
    # Note: We must reconstruct the simulator machinery here or pass it in. 
    # For purity of functional separation, we instantiate a lightweight simulator 
    # with fixed lens parameters corresponding to the problem setup.
    
    cosmo = FlatLambdaCDM(name="cosmo_fwd")
    # Fixed lens parameters (assumed known for this inversion task)
    lens = SIE(
        cosmology=cosmo, z_l=0.5, z_s=1.0, x0=0.0, y0=0.0, q=0.7, phi=0.5, Rein=1.5, name="lens_fwd"
    )
    
    # Create a source object wrapping x
    source = Pixelated(
        image=x, x0=0.0, y0=0.0, pixelscale=0.04, name="source_fwd"
    )
    
    sim = LensSource(
        lens=lens, source=source, pixelscale=0.05, pixels_x=128, upsample_factor=2, name="sim_fwd"
    )
    sim.to(device)
    
    return sim()

def evaluate_results(true_source: torch.Tensor, reconstructed_source: torch.Tensor, observation: torch.Tensor, device: torch.device):
    """
    Calculates metrics and generates visualization.
    """
    # 1. Generate predicted observation from reconstruction
    predicted_observation = forward_operator(reconstructed_source, device).detach()
    
    # 2. Metrics
    mse = torch.mean((reconstructed_source - true_source) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # SSIM calculation
    def calculate_ssim(img1, img2):
        C1 = (0.01 * 1.0)**2
        C2 = (0.03 * 1.0)**2
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.std()
        sigma2 = img2.std()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        return ssim

    ssim = calculate_ssim(reconstructed_source, true_source)
    
    print("="*30)
    print(f"RECONSTRUCTION METRICS (Source Plane):")
    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim.item():.4f}")
    print("="*30)
    
    # 3. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(true_source.cpu().numpy(), origin="lower", cmap="inferno")
    axes[0, 0].set_title("Ground Truth Source")
    
    axes[0, 1].imshow(reconstructed_source.cpu().numpy(), origin="lower", cmap="inferno")
    axes[0, 1].set_title(f"Reconstructed Source\nPSNR: {psnr.item():.2f} dB")
    
    axes[1, 0].imshow(observation.cpu().numpy(), origin="lower", cmap="inferno")
    axes[1, 0].set_title("Observation (Lensed)")
    
    axes[1, 1].imshow(predicted_observation.cpu().numpy(), origin="lower", cmap="inferno")
    axes[1, 1].set_title("Model Prediction")
    
    plt.tight_layout()
    plt.savefig("reconstruction_result.png")
    print("Result saved to reconstruction_result.png")
