import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import warnings

# --- Dependencies Check ---
try:
    import ehtim as eh
    import ehtim.const_def as ehc
    from ehtim.observing.obs_helpers import ftmatrix
except ImportError:
    print("Error: ehtim package not found. Please install it.")
    sys.exit(1)

try:
    from torchkbnufft import KbNufft
except ImportError:
    KbNufft = None


# ==============================================================================
# RealNVP Model Components
# ==============================================================================

class ActNorm(nn.Module):
    def __init__(self, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1,))
        self.log_scale_inv = nn.Parameter(torch.zeros(1,))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input, inv_init=False):
        with torch.no_grad():
            mean = input.mean().reshape((1,))
            std = input.std().reshape((1,))
            if inv_init:
                self.loc.data.copy_(torch.zeros_like(mean))
                self.log_scale_inv.data.copy_(torch.zeros_like(std))
            else:
                self.loc.data.copy_(-mean)
                self.log_scale_inv.data.copy_(torch.log(std + 1e-6))

    def forward(self, input):
        _, in_dim = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        scale_inv = torch.exp(self.log_scale_inv)
        log_abs = -self.log_scale_inv
        logdet = in_dim * torch.sum(log_abs)
        if self.logdet:
            return (1.0 / scale_inv) * (input + self.loc), logdet
        else:
            return (1.0 / scale_inv) * (input + self.loc)

    def reverse(self, output):
        _, in_dim = output.shape
        if self.initialized.item() == 0:
            self.initialize(output, inv_init=True)
            self.initialized.fill_(1)
        scale_inv = torch.exp(self.log_scale_inv)
        log_abs = -self.log_scale_inv
        logdet = -in_dim * torch.sum(log_abs)
        if self.logdet:
            return output * scale_inv - self.loc, logdet
        else:
            return output * scale_inv - self.loc


class ZeroFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(out_dim,))

    def forward(self, input):
        out = self.fc(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, ndim, seqfrac=4, affine=True, batch_norm=True):
        super().__init__()
        self.affine = affine
        self.batch_norm = batch_norm
        hidden_dim = max(1, int(ndim / (2 * seqfrac)))
        in_features = ndim - ndim // 2
        out_features = 2 * (ndim // 2) if self.affine else ndim // 2

        if batch_norm:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(hidden_dim, eps=1e-2, affine=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(hidden_dim, eps=1e-2, affine=True),
                ZeroFC(hidden_dim, out_features),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[3].weight.data.normal_(0, 0.05)
            self.net[3].bias.data.zero_()
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ZeroFC(hidden_dim, out_features),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[2].weight.data.normal_(0, 0.05)
            self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s0, t = self.net(in_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            out_b = (in_b + t) * s
            logdet = torch.sum(log_s.view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s0, t = self.net(out_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            in_b = out_b / s - t
            logdet = -torch.sum(log_s.view(output.shape[0], -1), 1)
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
            logdet = None
        return torch.cat([out_a, in_b], 1), logdet


class Flow(nn.Module):
    def __init__(self, ndim, affine=True, seqfrac=4, batch_norm=True):
        super().__init__()
        self.actnorm = ActNorm()
        self.actnorm2 = ActNorm()
        self.coupling = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine, batch_norm=batch_norm)
        self.coupling2 = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine, batch_norm=batch_norm)
        self.ndim = ndim

    def forward(self, input):
        logdet = 0
        out, det1 = self.actnorm(input)
        out, det2 = self.coupling(out)
        out = out[:, np.arange(self.ndim - 1, -1, -1)]
        out, det3 = self.actnorm2(out)
        out, det4 = self.coupling2(out)
        out = out[:, np.arange(self.ndim - 1, -1, -1)]
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        logdet = logdet + det3
        if det4 is not None:
            logdet = logdet + det4
        return out, logdet

    def reverse(self, output):
        logdet = 0
        input = output[:, np.arange(self.ndim - 1, -1, -1)]
        input, det1 = self.coupling2.reverse(input)
        input, det2 = self.actnorm2.reverse(input)
        input = input[:, np.arange(self.ndim - 1, -1, -1)]
        input, det3 = self.coupling.reverse(input)
        input, det4 = self.actnorm.reverse(input)
        if det1 is not None:
            logdet = logdet + det1
        logdet = logdet + det2
        if det3 is not None:
            logdet = logdet + det3
        logdet = logdet + det4
        return input, logdet


def Order_inverse(order):
    order_inv = []
    for k in range(len(order)):
        for i in range(len(order)):
            if order[i] == k:
                order_inv.append(i)
    return np.array(order_inv)


class RealNVP(nn.Module):
    def __init__(self, ndim, n_flow, affine=True, seqfrac=4, permute='random', batch_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.orders = []
        for i in range(n_flow):
            self.blocks.append(Flow(ndim, affine=affine, seqfrac=seqfrac, batch_norm=batch_norm))
            if permute == 'random':
                self.orders.append(np.random.RandomState(seed=i).permutation(ndim))
            elif permute == 'reverse':
                self.orders.append(np.arange(ndim - 1, -1, -1))
            else:
                self.orders.append(np.arange(ndim))
        self.inverse_orders = []
        for i in range(n_flow):
            self.inverse_orders.append(Order_inverse(self.orders[i]))

    def forward(self, input):
        logdet = 0
        out = input
        for i in range(len(self.blocks)):
            out, det = self.blocks[i](out)
            logdet = logdet + det
            out = out[:, self.orders[i]]
        return out, logdet

    def reverse(self, out):
        logdet = 0
        input = out
        for i in range(len(self.blocks) - 1, -1, -1):
            input = input[:, self.inverse_orders[i]]
            input, det = self.blocks[i].reverse(input)
            logdet = logdet + det
        return input, logdet


# ==============================================================================
# Geometric Model
# ==============================================================================

class SimpleCrescentNuisanceFloor_Param2Img(nn.Module):
    def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0], asym_range=[1e-3, 0.99],
                 width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2],
                 crescent_flux_range=[1e-3, 2.0], shift_range=[-200.0, 200.0], sigma_range=[1.0, 100.0],
                 gaussian_scale_range=[1e-3, 2.0], flux_flag=False):
        super().__init__()
        self.n_gaussian = n_gaussian
        self.fov = fov
        self.r_range = r_range
        self.asym_range = asym_range
        self.width_range = width_range
        self.floor_range = floor_range
        self.flux_range = flux_range
        self.crescent_flux_range = crescent_flux_range
        self.shift_range = shift_range
        self.sigma_range = sigma_range
        self.gaussian_scale_range = gaussian_scale_range
        self.flux_flag = flux_flag
        if self.flux_flag:
            self.nparams = 5 + 6 * n_gaussian + 2
        else:
            self.nparams = 4 + 6 * n_gaussian + 2

        self.eps = 1e-4
        self.gap = 1.0 / npix
        xs = torch.arange(-1 + self.gap, 1, 2 * self.gap)
        grid_y, grid_x = torch.meshgrid(-xs, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)
        self.register_buffer('grid_r', torch.sqrt(grid_x ** 2 + grid_y ** 2))
        self.register_buffer('grid_theta', torch.atan2(grid_y, grid_x))
        self.npix = npix

    def compute_features(self, params):
        r = self.r_range[0] / (0.5 * self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (
                    self.r_range[1] - self.r_range[0]) / (0.5 * self.fov)
        sigma = self.width_range[0] / (0.5 * self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (
                    self.width_range[1] - self.width_range[0]) / (0.5 * self.fov)
        s = self.asym_range[0] + params[:, 2].unsqueeze(-1).unsqueeze(-1) * (self.asym_range[1] - self.asym_range[0])
        eta = 181 / 180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)

        nuisance_scale = []
        sigma_x_list = []
        sigma_y_list = []
        theta_list = []
        nuisance_x = []
        nuisance_y = []
        for k in range(self.n_gaussian):
            x_shift = self.shift_range[0] / (0.5 * self.fov) + params[:, 4 + k * 6].unsqueeze(-1).unsqueeze(-1) * (
                        self.shift_range[1] - self.shift_range[0]) / (0.5 * self.fov)
            y_shift = self.shift_range[0] / (0.5 * self.fov) + params[:, 5 + k * 6].unsqueeze(-1).unsqueeze(-1) * (
                        self.shift_range[1] - self.shift_range[0]) / (0.5 * self.fov)
            scale = self.gaussian_scale_range[0] + params[:, 6 + k * 6].unsqueeze(-1).unsqueeze(-1) * (
                        self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
            sigma_x = self.sigma_range[0] / (0.5 * self.fov) + params[:, 7 + k * 6].unsqueeze(-1).unsqueeze(-1) * (
                        self.sigma_range[1] - self.sigma_range[0]) / (0.5 * self.fov)
            sigma_y = self.sigma_range[0] / (0.5 * self.fov) + params[:, 8 + k * 6].unsqueeze(-1).unsqueeze(-1) * (
                        self.sigma_range[1] - self.sigma_range[0]) / (0.5 * self.fov)
            theta = 181 / 180 * 0.5 * np.pi * params[:, 9 + k * 6].unsqueeze(-1).unsqueeze(-1)

            nuisance_x.append(x_shift)
            nuisance_y.append(y_shift)
            nuisance_scale.append(scale)
            sigma_x_list.append(sigma_x)
            sigma_y_list.append(sigma_y)
            theta_list.append(theta)

        if self.flux_flag:
            total_flux = self.flux_range[0] + (self.flux_range[1] - self.flux_range[0]) * params[:,
                                                                                          4 + self.n_gaussian * 6].unsqueeze(
                -1).unsqueeze(-1)
            floor = self.floor_range[0] + (self.floor_range[1] - self.floor_range[0]) * params[:,
                                                                                        5 + self.n_gaussian * 6].unsqueeze(
                -1).unsqueeze(-1)
            crescent_flux = self.crescent_flux_range[0] + (
                        self.crescent_flux_range[1] - self.crescent_flux_range[0]) * params[:,
                                                                                     6 + self.n_gaussian * 6].unsqueeze(
                -1).unsqueeze(-1)
            return r, sigma, s, eta, total_flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
        else:
            floor = self.floor_range[0] + (self.floor_range[1] - self.floor_range[0]) * params[:,
                                                                                        4 + self.n_gaussian * 6].unsqueeze(
                -1).unsqueeze(-1)
            crescent_flux = self.crescent_flux_range[0] + (
                        self.crescent_flux_range[1] - self.crescent_flux_range[0]) * params[:,
                                                                                     5 + self.n_gaussian * 6].unsqueeze(
                -1).unsqueeze(-1)
            return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list

    def forward(self, params):
        if self.flux_flag:
            r, sigma, s, eta, flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(
                params)
        else:
            r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(
                params)

        ring = torch.exp(-0.5 * (self.grid_r - r) ** 2 / (sigma) ** 2)
        S = 1 + s * torch.cos(self.grid_theta - eta)
        crescent = S * ring
        disk = 0.5 * (1 + torch.erf((r - self.grid_r) / (np.sqrt(2) * sigma)))

        crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        crescent = crescent_flux * ((1 - floor) * crescent + floor * disk)

        for k in range(self.n_gaussian):
            x_c = self.grid_x - nuisance_x[k]
            y_c = self.grid_y - nuisance_y[k]
            x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
            y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
            delta = 0.5 * (x_rot ** 2 / sigma_x_list[k] ** 2 + y_rot ** 2 / sigma_y_list[k] ** 2)
            nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
            nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
            nuisance_now = nuisance_scale[k] * nuisance_now
            crescent = crescent + nuisance_now

        if self.flux_flag:
            crescent = flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)
        else:
            crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1) + self.eps)

        return crescent


# ==============================================================================
# Helper Functions
# ==============================================================================

def torch_complex_mul(x, y):
    xy_real = x[:, :, 0:1] * y[0:1] - x[:, :, 1::] * y[1::]
    xy_imag = x[:, :, 0:1] * y[1::] + x[:, :, 1::] * y[0:1]
    return torch.cat([xy_real, xy_imag], -2)


def torch_complex_matmul(x, F):
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)


# ==============================================================================
# 1. load_and_preprocess_data
# ==============================================================================

def load_and_preprocess_data(obspath, npix, fov, prior_fwhm, snrcut=0.0, ttype='direct'):
    """
    Load observation data from uvfits file and compute all preprocessing.
    Returns all data required for forward model and inversion.
    """
    obs = eh.obsdata.load_uvfits(obspath)

    # Prior image setup
    flux_const = np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp'])
    prior_fwhm_rad = prior_fwhm * eh.RADPERUAS
    fov_rad = fov * eh.RADPERUAS
    zbl = flux_const

    prior = eh.image.make_square(obs, npix, fov_rad)
    prior = prior.add_tophat(zbl, prior_fwhm_rad / 2.0)
    prior = prior.blur_circ(obs.res())
    simim = prior.copy()
    simim.ra = obs.ra
    simim.dec = obs.dec
    simim.rf = obs.rf

    # Extract observation parameters
    obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
    uv = np.hstack((obs_data['u'].reshape(-1, 1), obs_data['v'].reshape(-1, 1)))
    vu = np.hstack((obs_data['v'].reshape(-1, 1), obs_data['u'].reshape(-1, 1)))

    # Compute DFT matrix
    dft_mat = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)
    dft_mat = np.expand_dims(dft_mat.T, -1)
    dft_mat = np.concatenate([dft_mat.real, dft_mat.imag], -1)
    dft_mat = torch.tensor(dft_mat, dtype=torch.float32)

    # Compute NFFT trajectory
    vu_scaled = np.array(vu * simim.psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T).unsqueeze(0)

    # Pulse factor (simplified)
    fft_pad_factor = ehc.FFT_PAD_DEFAULT
    p_rad = ehc.GRIDDER_P_RAD_DEFAULT
    npad = int(fft_pad_factor * np.max((simim.xdim, simim.ydim)))
    try:
        from ehtim.observing.obs_helpers import NFFTInfo
        nfft_info_vis = NFFTInfo(simim.xdim, simim.ydim, simim.psize, simim.pulse, npad, p_rad, uv)
        pulsefac_vis = nfft_info_vis.pulsefac
    except:
        pulsefac_vis = np.ones(uv.shape[0], dtype=np.complex128)

    pulsefac_vis_torch = torch.tensor(np.concatenate([np.expand_dims(pulsefac_vis.real, 0),
                                                      np.expand_dims(pulsefac_vis.imag, 0)], 0))

    # Add closure phases
    if snrcut > 0:
        obs.add_cphase(count='min-cut0bl', uv_min=.1e9, snrcut=snrcut)
    else:
        obs.add_cphase(count='min-cut0bl', uv_min=.1e9)

    # Build cphase map
    cphase_map = np.zeros((len(obs.cphase['time']), 3))
    zero_symbol = 100000
    for k1 in range(cphase_map.shape[0]):
        for k2 in list(np.where(obs.data['time'] == obs.cphase['time'][k1])[0]):
            if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = k2
                if k2 == 0: cphase_map[k1, 0] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = -k2
                if k2 == 0: cphase_map[k1, 0] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = k2
                if k2 == 0: cphase_map[k1, 1] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = -k2
                if k2 == 0: cphase_map[k1, 1] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = k2
                if k2 == 0: cphase_map[k1, 2] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = -k2
                if k2 == 0: cphase_map[k1, 2] = -zero_symbol

    cphase_ind1 = np.abs(cphase_map[:, 0]).astype(np.int64)
    cphase_ind1[cphase_ind1 == zero_symbol] = 0
    cphase_ind2 = np.abs(cphase_map[:, 1]).astype(np.int64)
    cphase_ind2[cphase_ind2 == zero_symbol] = 0
    cphase_ind3 = np.abs(cphase_map[:, 2]).astype(np.int64)
    cphase_ind3[cphase_ind3 == zero_symbol] = 0
    cphase_sign1 = np.sign(cphase_map[:, 0])
    cphase_sign2 = np.sign(cphase_map[:, 1])
    cphase_sign3 = np.sign(cphase_map[:, 2])

    cphase_ind_list = [torch.tensor(cphase_ind1), torch.tensor(cphase_ind2), torch.tensor(cphase_ind3)]
    cphase_sign_list = [torch.tensor(cphase_sign1), torch.tensor(cphase_sign2), torch.tensor(cphase_sign3)]

    # Add closure amplitudes
    if snrcut > 0:
        obs.add_camp(debias=True, count='min', snrcut=snrcut)
        obs.add_logcamp(debias=True, count='min', snrcut=snrcut)
    else:
        obs.add_camp(debias=True, count='min')
        obs.add_logcamp(debias=True, count='min')

    # Build camp map
    camp_map = np.zeros((len(obs.camp['time']), 6))
    zero_symbol = 10000
    for k1 in range(camp_map.shape[0]):
        for k2 in list(np.where(obs.data['time'] == obs.camp['time'][k1])[0]):
            if obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t2'][k1]:
                camp_map[k1, 0] = k2
                if k2 == 0: camp_map[k1, 0] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t2'][k1]:
                camp_map[k1, 0] = -k2
                if k2 == 0: camp_map[k1, 0] = -zero_symbol
            elif obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t3'][k1]:
                camp_map[k1, 1] = k2
                if k2 == 0: camp_map[k1, 1] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t3'][k1]:
                camp_map[k1, 1] = -k2
                if k2 == 0: camp_map[k1, 1] = -zero_symbol
            elif obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 2] = k2
                if k2 == 0: camp_map[k1, 2] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 2] = -k2
                if k2 == 0: camp_map[k1, 2] = -zero_symbol
            elif obs.data['t1'][k2] == obs.camp['t2'][k1] and obs.data['t2'][k2] == obs.camp['t3'][k1]:
                camp_map[k1, 3] = k2
                if k2 == 0: camp_map[k1, 3] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t2'][k1] and obs.data['t1'][k2] == obs.camp['t3'][k1]:
                camp_map[k1, 3] = -k2
                if k2 == 0: camp_map[k1, 3] = -zero_symbol
            elif obs.data['t1'][k2] == obs.camp['t2'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 4] = k2
                if k2 == 0: camp_map[k1, 4] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t2'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 4] = -k2
                if k2 == 0: camp_map[k1, 4] = -zero_symbol
            elif obs.data['t1'][k2] == obs.camp['t3'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 5] = k2
                if k2 == 0: camp_map[k1, 5] = zero_symbol
            elif obs.data['t2'][k2] == obs.camp['t3'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
                camp_map[k1, 5] = -k2
                if k2 == 0: camp_map[k1, 5] = -zero_symbol

    camp_ind1 = np.abs(camp_map[:, 0]).astype(np.int64)
    camp_ind1[camp_ind1 == zero_symbol] = 0
    camp_ind2 = np.abs(camp_map[:, 5]).astype(np.int64)
    camp_ind2[camp_ind2 == zero_symbol] = 0
    camp_ind3 = np.abs(camp_map[:, 2]).astype(np.int64)
    camp_ind3[camp_ind3 == zero_symbol] = 0
    camp_ind4 = np.abs(camp_map[:, 3]).astype(np.int64)
    camp_ind4[camp_ind4 == zero_symbol] = 0

    camp_ind_list = [torch.tensor(camp_ind1), torch.tensor(camp_ind2), torch.tensor(camp_ind3), torch.tensor(camp_ind4)]

    # Ground truth data
    vis_true = torch.tensor(np.concatenate([np.expand_dims(obs.data['vis'].real, 0),
                                            np.expand_dims(obs.data['vis'].imag, 0)], 0), dtype=torch.float32)
    data_arr = obs.unpack(['amp'], debias=True)
    visamp_true = torch.tensor(np.array(data_arr['amp']), dtype=torch.float32)
    cphase_true = torch.tensor(np.array(obs.cphase['cphase']), dtype=torch.float32)
    logcamp_true = torch.tensor(np.array(obs.logcamp['camp']), dtype=torch.float32)

    # Sigmas
    sigma_vis = torch.tensor(obs.data['sigma'], dtype=torch.float32)
    sigma_cphase = torch.tensor(obs.cphase['sigmacp'], dtype=torch.float32)
    sigma_logcamp = torch.tensor(obs.logcamp['sigmaca'], dtype=torch.float32)

    preprocessed_data = {
        'obs': obs,
        'simim': simim,
        'flux_const': flux_const,
        'dft_mat': dft_mat,
        'ktraj_vis': ktraj_vis,
        'pulsefac_vis_torch': pulsefac_vis_torch,
        'cphase_ind_list': cphase_ind_list,
        'cphase_sign_list': cphase_sign_list,
        'camp_ind_list': camp_ind_list,
        'vis_true': vis_true,
        'visamp_true': visamp_true,
        'cphase_true': cphase_true,
        'logcamp_true': logcamp_true,
        'sigma_vis': sigma_vis,
        'sigma_cphase': sigma_cphase,
        'sigma_logcamp': sigma_logcamp,
        'npix': npix,
        'fov': fov,
    }

    return preprocessed_data


# ==============================================================================
# 2. forward_operator
# ==============================================================================

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device):
    """
    Compute interferometric observables from image x.
    Returns vis_torch, vis_amp, cphase, logcamp as numpy arrays.
    """
    eps = 1e-16

    # Ensure tensors are on device
    F = dft_mat.to(device=device)
    cphase_ind1 = cphase_ind_list[0].to(device=device)
    cphase_ind2 = cphase_ind_list[1].to(device=device)
    cphase_ind3 = cphase_ind_list[2].to(device=device)
    cphase_sign1 = cphase_sign_list[0].to(device=device)
    cphase_sign2 = cphase_sign_list[1].to(device=device)
    cphase_sign3 = cphase_sign_list[2].to(device=device)
    camp_ind1 = camp_ind_list[0].to(device=device)
    camp_ind2 = camp_ind_list[1].to(device=device)
    camp_ind3 = camp_ind_list[2].to(device=device)
    camp_ind4 = camp_ind_list[3].to(device=device)

    # Ensure x is on device and properly shaped
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device=device)
    x = torch.reshape(x, (-1, npix * npix)).type(torch.float32)

    # Compute visibilities via DFT
    vis_torch = torch_complex_matmul(x, F)

    # Visibility amplitude
    vis_amp = torch.sqrt((vis_torch[:, 0, :]) ** 2 + (vis_torch[:, 1, :]) ** 2 + eps)

    # Closure phase
    vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
    vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
    vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)

    ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
    ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
    ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
    cphase = (cphase_sign1 * ang1 + cphase_sign2 * ang2 + cphase_sign3 * ang3) * 180 / np.pi

    # Log closure amplitude
    vis12_torch = torch.index_select(vis_torch, -1, camp_ind1)
    vis12_amp = torch.sqrt((vis12_torch[:, 0, :]) ** 2 + (vis12_torch[:, 1, :]) ** 2 + eps)
    vis34_torch = torch.index_select(vis_torch, -1, camp_ind2)
    vis34_amp = torch.sqrt((vis34_torch[:, 0, :]) ** 2 + (vis34_torch[:, 1, :]) ** 2 + eps)
    vis14_torch = torch.index_select(vis_torch, -1, camp_ind3)
    vis14_amp = torch.sqrt((vis14_torch[:, 0, :]) ** 2 + (vis14_torch[:, 1, :]) ** 2 + eps)
    vis23_torch = torch.index_select(vis_torch, -1, camp_ind4)
    vis23_amp = torch.sqrt((vis23_torch[:, 0, :]) ** 2 + (vis23_torch[:, 1, :]) ** 2 + eps)

    logcamp = torch.log(vis12_amp) + torch.log(vis34_amp) - torch.log(vis14_amp) - torch.log(vis23_amp)

    return vis_torch, vis_amp, cphase, logcamp


# ==============================================================================
# 3. run_inversion
# ==============================================================================

def run_inversion(preprocessed_data, n_flow=16, n_gaussian=2, n_epoch=10, n_batch=64, lr=1e-4, clip=1e-4,
                  data_product='cphase_logcamp', divergence_type='alpha', alpha_divergence=1.0,
                  start_order=4, decay_rate=2000, beta=0.0, logdet_weight=1.0, device=None):
    """
    Run the deep probabilistic imaging inversion.
    Returns the trained model state dict and final loss.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npix = preprocessed_data['npix']
    fov = preprocessed_data['fov']
    flux_const = preprocessed_data['flux_const']

    # Move data to device
    dft_mat = preprocessed_data['dft_mat'].to(device)
    cphase_ind_list = [t.to(device) for t in preprocessed_data['cphase_ind_list']]
    cphase_sign_list = [t.to(device) for t in preprocessed_data['cphase_sign_list']]
    camp_ind_list = [t.to(device) for t in preprocessed_data['camp_ind_list']]

    vis_true = preprocessed_data['vis_true'].to(device)
    visamp_true = preprocessed_data['visamp_true'].to(device)
    cphase_true = preprocessed_data['cphase_true'].to(device)
    logcamp_true = preprocessed_data['logcamp_true'].to(device)

    sigma_vis = preprocessed_data['sigma_vis'].to(device)
    sigma_cphase = preprocessed_data['sigma_cphase'].to(device)
    sigma_logcamp = preprocessed_data['sigma_logcamp'].to(device)

    # Setup model
    flux_flag = data_product != 'cphase_logcamp'
    flux_range = [0.8 * flux_const, 1.2 * flux_const]
    r_range = [10.0, 40.0]

    img_converter = SimpleCrescentNuisanceFloor_Param2Img(
        npix, r_range=r_range, fov=fov, n_gaussian=n_gaussian,
        flux_flag=flux_flag, flux_range=flux_range
    ).to(device=device)

    nparams = img_converter.nparams
    params_generator = RealNVP(nparams, n_flow, affine=True, seqfrac=1 / 16,
                               permute='random', batch_norm=True).to(device)

    # Setup weights
    obs = preprocessed_data['obs']
    if data_product == 'vis':
        vis_weight = 1.0
        camp_weight = 0.0
        cphase_weight = 0.0
        visamp_weight = 0.0
        logdet_scale = 2.0 * logdet_weight / len(obs.data['vis'])
        scale_factor = 1.0 / len(obs.data['vis'])
    else:  # cphase_logcamp
        vis_weight = 0.0
        camp_weight = 1.0
        cphase_weight = len(obs.cphase['cphase']) / len(obs.camp['camp'])
        visamp_weight = 0.0
        logdet_scale = 2.0 * logdet_weight / len(obs.camp['camp'])
        scale_factor = 1.0 / len(obs.camp['camp'])

    optimizer = optim.Adam(params_generator.parameters(), lr=lr)

    if beta == 0:
        alpha_div = alpha_divergence
    else:
        alpha_div = 1 - beta * scale_factor

    final_loss = 0.0
    for k in range(n_epoch):
        data_weight = min(10 ** (-start_order + k / decay_rate), 1.0)
        z_sample = torch.randn((n_batch, nparams)).to(device=device)

        params_samp, logdet = params_generator.reverse(z_sample)
        params = torch.sigmoid(params_samp)
        img = img_converter.forward(params)

        det_sigmoid = torch.sum(-params_samp - 2 * torch.nn.Softplus()(-params_samp), -1)
        logdet = logdet + det_sigmoid

        # Forward model
        vis, visamp, cphase, logcamp = forward_operator(
            img, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device
        )

        # Compute losses
        if vis_weight == 0:
            loss_cphase = 2.0 * torch.mean((1 - torch.cos((cphase_true - cphase) * np.pi / 180)) / (
                        sigma_cphase * np.pi / 180) ** 2, 1) if cphase_weight > 0 else 0
            loss_camp = torch.mean((logcamp_true - logcamp) ** 2 / (sigma_logcamp) ** 2, 1) if camp_weight > 0 else 0
            loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase
            loss_data = 0.5 * loss_data / scale_factor
        else:
            loss_vis = torch.mean(((vis_true[0] - vis[:, 0]) ** 2 + (vis_true[1] - vis[:, 1]) ** 2) / (sigma_vis) ** 2,
                                  1)
            loss_data = vis_weight * loss_vis
            loss_data = 0.5 * loss_data / scale_factor

        logprob = -logdet - 0.5 * torch.sum(z_sample ** 2, 1)
        loss = data_weight * loss_data + logprob

        if divergence_type == 'KL' or alpha_div == 1:
            loss = torch.mean(scale_factor * loss)
        else:
            if beta == 0:
                rej_weights = nn.Softmax(dim=0)(-(1 - alpha_div) * loss).detach()
            else:
                rej_weights = nn.Softmax(dim=0)(-beta * scale_factor * loss).detach()
            loss = torch.sum(rej_weights * scale_factor * loss)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params_generator.parameters(), clip)
        optimizer.step()

        final_loss = loss.item()
        if k % 100 == 0 or k == n_epoch - 1:
            print(f"Epoch {k}: Loss {final_loss:.6f}")

    result = {
        'model_state_dict': params_generator.state_dict(),
        'final_loss': final_loss,
        'nparams': nparams,
        'n_flow': n_flow,
    }

    return result


# ==============================================================================
# 4. evaluate_results
# ==============================================================================

def evaluate_results(result, preprocessed_data, n_samples=100, n_gaussian=2, device=None):
    """
    Evaluate the trained model by generating sample images and computing statistics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npix = preprocessed_data['npix']
    fov = preprocessed_data['fov']
    flux_const = preprocessed_data['flux_const']

    nparams = result['nparams']
    n_flow = result['n_flow']

    # Reconstruct models
    flux_flag = False
    flux_range = [0.8 * flux_const, 1.2 * flux_const]
    r_range = [10.0, 40.0]

    img_converter = SimpleCrescentNuisanceFloor_Param2Img(
        npix, r_range=r_range, fov=fov, n_gaussian=n_gaussian,
        flux_flag=flux_flag, flux_range=flux_range
    ).to(device=device)

    params_generator = RealNVP(nparams, n_flow, affine=True, seqfrac=1 / 16,
                               permute='random', batch_norm=True).to(device)
    params_generator.load_state_dict(result['model_state_dict'])
    params_generator.eval()

    # Generate samples
    with torch.no_grad():
        z_sample = torch.randn((n_samples, nparams)).to(device=device)
        params_samp, _ = params_generator.reverse(z_sample)
        params = torch.sigmoid(params_samp)
        imgs = img_converter.forward(params)

    imgs_np = imgs.cpu().numpy()

    # Compute statistics
    mean_img = np.mean(imgs_np, axis=0)
    std_img = np.std(imgs_np, axis=0)
    total_flux_samples = np.sum(imgs_np, axis=(1, 2))

    # Compute data fit for mean image
    dft_mat = preprocessed_data['dft_mat'].to(device)
    cphase_ind_list = [t.to(device) for t in preprocessed_data['cphase_ind_list']]
    cphase_sign_list = [t.to(device) for t in preprocessed_data['cphase_sign_list']]
    camp_ind_list = [t.to(device) for t in preprocessed_data['camp_ind_list']]

    mean_img_tensor = torch.tensor(mean_img, dtype=torch.float32).unsqueeze(0).to(device)
    vis, visamp, cphase, logcamp = forward_operator(
        mean_img_tensor, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device
    )

    cphase_true = preprocessed_data['cphase_true'].to(device)
    logcamp_true = preprocessed_data['logcamp_true'].to(device)
    sigma_cphase = preprocessed_data['sigma_cphase'].to(device)
    sigma_logcamp = preprocessed_data['sigma_logcamp'].to(device)

    cphase_residual = torch.mean(
        (1 - torch.cos((cphase_true - cphase) * np.pi / 180)) / (sigma_cphase * np.pi / 180) ** 2).item()
    logcamp_residual = torch.mean((logcamp_true - logcamp) ** 2 / sigma_logcamp ** 2).item()

    evaluation = {
        'mean_image': mean_img,
        'std_image': std_img,
        'sample_images': imgs_np,
        'total_flux_mean': np.mean(total_flux_samples),
        'total_flux_std': np.std(total_flux_samples),
        'cphase_chi2': cphase_residual,
        'logcamp_chi2': logcamp_residual,
        'final_loss': result['final_loss'],
    }

    print("\n=== Evaluation Results ===")
    print(f"Mean image shape: {mean_img.shape}")
    print(f"Total flux: {evaluation['total_flux_mean']:.6f} +/- {evaluation['total_flux_std']:.6f}")
    print(f"Closure phase chi2: {evaluation['cphase_chi2']:.4f}")
    print(f"Log closure amplitude chi2: {evaluation['logcamp_chi2']:.4f}")
    print(f"Final training loss: {evaluation['final_loss']:.6f}")

    return evaluation


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging for Interferometry (Refactored)")
    parser.add_argument("--cuda", default=0, type=int, help="cuda index in use")
    parser.add_argument("--obspath", type=str, default=None, help="EHT observation file path")
    parser.add_argument("--save_path", default='./checkpoint', type=str, help="file save path")
    parser.add_argument("--npix", default=64, type=int, help="image shape (pixels)")
    parser.add_argument("--n_gaussian", default=2, type=int, help="number of additional nuisance gaussian")
    parser.add_argument("--fov", default=120, type=float, help="field of view of the image in micro-arcsecond")
    parser.add_argument("--prior_fwhm", default=80, type=float, help="fwhm of image prior in micro-arcsecond")
    parser.add_argument("--n_flow", default=16, type=int, help="number of flows in RealNVP")
    parser.add_argument("--logdet", default=1.0, type=float, help="logdet weight")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--clip", default=1e-4, type=float, help="gradient clip")
    parser.add_argument("--data_product", default='cphase_logcamp', type=str, help="data product")
    parser.add_argument("--divergence_type", default='alpha', type=str, help="KL or alpha")
    parser.add_argument("--alpha_divergence", default=1.0, type=float, help="alpha divergence param")
    parser.add_argument("--start_order", default=4, type=float, help="start order")
    parser.add_argument("--decay_rate", default=2000, type=float, help="decay rate")
    parser.add_argument("--n_epoch", default=10, type=int, help="number of epochs")
    parser.add_argument("--beta", default=0.0, type=float, help="beta param")

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Check if observation path provided
    if args.obspath is None or not os.path.exists(args.obspath):
        print("No valid observation file provided. Running in demo mode with synthetic data.")

        # Create synthetic observation for testing
        npix = args.npix
        fov = args.fov
        n_gaussian = args.n_gaussian

        # Create a simple synthetic test
        print("\nCreating synthetic test data...")

        # Create minimal synthetic preprocessed data
        n_vis = 100
        n_cphase = 50
        n_camp = 30

        # Generate random DFT matrix
        dft_mat = torch.randn(npix * npix, n_vis, 2, dtype=torch.float32)

        # Generate random indices for closure quantities
        cphase_ind_list = [torch.randint(0, n_vis, (n_cphase,)) for _ in range(3)]
        cphase_sign_list = [torch.sign(torch.randn(n_cphase)) for _ in range(3)]
        camp_ind_list = [torch.randint(0, n_vis, (n_camp,)) for _ in range(4)]

        # Generate random ground truth
        vis_true = torch.randn(2, n_vis, dtype=torch.float32)
        visamp_true = torch.abs(torch.randn(n_vis, dtype=torch.float32))
        cphase_true = torch.randn(n_cphase, dtype=torch.float32) * 30
        logcamp_true = torch.randn(n_camp, dtype=torch.float32) * 0.1

        # Generate sigmas
        sigma_vis = torch.ones(n_vis, dtype=torch.float32) * 0.1
        sigma_cphase = torch.ones(n_cphase, dtype=torch.float32) * 5.0
        sigma_logcamp = torch.ones(n_camp, dtype=torch.float32) * 0.05

        # Create mock obs object with required attributes
        class MockObs:
            def __init__(self):
                self.data = {'vis': np.random.randn(n_vis) + 1j * np.random.randn(n_vis),
                             'sigma': np.ones(n_vis) * 0.1}
                self.cphase = {'cphase': np.random.randn(n_cphase) * 30,
                               'sigmacp': np.ones(n_cphase) * 5.0}
                self.camp = {'camp': np.random.randn(n_camp) * 0.5}
                self.logcamp = {'camp': np.random.randn(n_camp) * 0.1,
                                'sigmaca': np.ones(n_camp) * 0.05}

        preprocessed_data = {
            'obs': MockObs(),
            'simim': None,
            'flux_const': 1.0,
            'dft_mat': dft_mat,
            'ktraj_vis': torch.randn(1, 2, n_vis),
            'pulsefac_vis_torch': torch.ones(2, n_vis),
            'cphase_ind_list': cphase_ind_list,
            'cphase_sign_list': cphase_sign_list,
            'camp_ind_list': camp_ind_list,
            'vis_true': vis_true,
            'visamp_true': visamp_true,
            'cphase_true': cphase_true,
            'logcamp_true': logcamp_true,
            'sigma_vis': sigma_vis,
            'sigma_cphase': sigma_cphase,
            'sigma_logcamp': sigma_logcamp,
            'npix': npix,
            'fov': fov,
        }

        print("Synthetic data created.")

    else:
        # Load real data
        print(f"\nLoading observation from: {args.obspath}")
        preprocessed_data = load_and_preprocess_data(
            obspath=args.obspath,
            npix=args.npix,
            fov=args.fov,
            prior_fwhm=args.prior_fwhm,
            snrcut=0.0,
            ttype='direct'
        )
        print("Data loaded and preprocessed.")

    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Run inversion
    print("\nRunning inversion...")
    result = run_inversion(
        preprocessed_data=preprocessed_data,
        n_flow=args.n_flow,
        n_gaussian=args.n_gaussian,
        n_epoch=args.n_epoch,
        n_batch=64,
        lr=args.lr,
        clip=args.clip,
        data_product=args.data_product,
        divergence_type=args.divergence_type,
        alpha_divergence=args.alpha_divergence,
        start_order=args.start_order,
        decay_rate=args.decay_rate,
        beta=args.beta,
        logdet_weight=args.logdet,
        device=device
    )
    print("Inversion complete.")

    # Save model
    torch.save(result['model_state_dict'], f"{args.save_path}/final_model.pth")
    print(f"Model saved to {args.save_path}/final_model.pth")

    # Evaluate results
    print("\nEvaluating results...")
    evaluation = evaluate_results(
        result=result,
        preprocessed_data=preprocessed_data,
        n_samples=50,
        n_gaussian=args.n_gaussian,
        device=device
    )

    print("\nOPTIMIZATION_FINISHED_SUCCESSFULLY")