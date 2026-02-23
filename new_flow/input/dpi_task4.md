# dpi_task4.py
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
import torch.optim as optim
import pickle
import math
import argparse
import time
import sys
import copy
import warnings
import pandas as pd

# Check dependencies
try:
    import astropy.units as u
    import astropy.constants as consts
except ImportError:
    print("Error: astropy not found. Please install it.")
    sys.exit(1)

# --- RealNVP Model (Inlined) ---
class ActNorm(nn.Module):
    def __init__(self, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, ))
        self.log_scale_inv = nn.Parameter(torch.zeros(1, ))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input, inv_init=False):
        with torch.no_grad():
            mean = input.mean().reshape((1, ))
            std = input.std().reshape((1, ))
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
        self.scale = nn.Parameter(torch.zeros(out_dim, ))

    def forward(self, input):
        out = self.fc(input)
        out = out * torch.exp(self.scale * 3)
        return out

class AffineCoupling(nn.Module):
    def __init__(self, ndim, seqfrac=4, affine=True, batch_norm=True):
        super().__init__()
        self.affine = affine
        self.batch_norm = batch_norm
        if batch_norm:
            self.net = nn.Sequential(
                nn.Linear(ndim-ndim//2, int(ndim / (2*seqfrac))),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(int(ndim / (2*seqfrac)), eps=1e-2, affine=True),
                nn.Linear(int(ndim / (2*seqfrac)), int(ndim / (2*seqfrac))),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm1d(int(ndim / (2*seqfrac)), eps=1e-2, affine=True),
                ZeroFC(int(ndim / (2*seqfrac)), 2*(ndim // 2) if self.affine else ndim // 2),
            )
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.data.zero_()
            self.net[3].weight.data.normal_(0, 0.05)
            self.net[3].bias.data.zero_()
        else:
            self.net = nn.Sequential(
                nn.Linear(ndim-ndim//2, int(ndim / (2*seqfrac))),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(int(ndim / (2*seqfrac)), int(ndim / (2*seqfrac))),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ZeroFC(int(ndim / (2*seqfrac)), 2*(ndim // 2) if self.affine else ndim // 2),
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
        out = out[:, np.arange(self.ndim-1, -1, -1)]
        out, det3 = self.actnorm2(out)
        out, det4 = self.coupling2(out)
        out = out[:, np.arange(self.ndim-1, -1, -1)]
        logdet = logdet + det1
        if det2 is not None: logdet = logdet + det2
        logdet = logdet + det3
        if det4 is not None: logdet = logdet + det4
        return out, logdet

    def reverse(self, output):
        logdet = 0
        input = output[:, np.arange(self.ndim-1, -1, -1)]
        input, det1 = self.coupling2.reverse(input)
        input, det2 = self.actnorm2.reverse(input)
        input = input[:, np.arange(self.ndim-1, -1, -1)]
        input, det3 = self.coupling.reverse(input)
        input, det4 = self.actnorm.reverse(input)
        if det1 is not None: logdet = logdet + det1
        logdet = logdet + det2
        if det3 is not None: logdet = logdet + det3
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
                self.orders.append(np.arange(ndim-1, -1, -1))
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
        for i in range(len(self.blocks)-1, -1, -1):
            input = input[:, self.inverse_orders[i]]
            input, det = self.blocks[i].reverse(input)
            logdet = logdet + det
        return input, logdet

# --- Orbit Helpers (Inlined) ---
def _mikkola_solver_torch(manom, ecc):
    alpha = (1.0 - ecc) / ((4.0 * ecc) + 0.5)
    beta = (0.5 * manom) / ((4.0 * ecc) + 0.5)

    aux = torch.sqrt(beta**2.0 + alpha**3.0)
    z = torch.abs(beta + aux)**(1.0/3.0)

    s0 = z - (alpha/z)
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + ecc)
    e0 = manom + (ecc * (3.0*s1 - 4.0*(s1**3.0)))

    se0 = torch.sin(e0)
    ce0 = torch.cos(e0)

    f = e0-ecc*se0-manom
    f1 = 1.0-ecc*ce0
    f2 = ecc*se0
    f3 = ecc*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+(1.0/6.0)*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+(1.0/6.0)*f3*u3*u3+(1.0/24.0)*f4*(u3**3.0))

    return (e0 + u4)

def _mikkola_solver_wrapper_torch(manom, ecc):
    eanom1 = _mikkola_solver_torch(manom, ecc)
    manom2 = (2.0 * np.pi) - manom
    eanom2 = _mikkola_solver_torch(manom2, ecc)
    eanom2 = (2.0 * np.pi) - eanom2

    eanom = torch.where(manom > np.pi, eanom2, eanom1)
    return eanom

def _newton_solver_torch(manom, ecc, tolerance=1e-9, max_iter=10, eanom0=None):
    if eanom0 is None:
        eanom = 1.0 * manom
    else:
        eanom = 1.0 * eanom0

    eanom = eanom - (eanom - (ecc * torch.sin(eanom)) - manom) / (1.0 - (ecc * torch.cos(eanom)))
    diff = (eanom - (ecc * torch.sin(eanom)) - manom) / (1.0 - (ecc * torch.cos(eanom)))
    
    niter = 0
    while niter <= max_iter:
        diff = (eanom - (ecc * torch.sin(eanom)) - manom) / (1.0 - (ecc * torch.cos(eanom)))
        eanom = eanom - diff
        niter += 1

    diff = (eanom - (ecc * torch.sin(eanom)) - manom) / (1.0 - (ecc * torch.cos(eanom)))
    abs_diff = torch.abs(diff)
    return eanom, abs_diff

def _calc_ecc_anom_torch(manom, ecc, tolerance=1e-9, max_iter=100):
    eanom = np.full(np.shape(manom), np.nan)
    ecc_zero = ecc == 0.0
    ecc_high = ecc >= 0.95

    eanom, abs_diff = _newton_solver_torch(manom, ecc, tolerance=tolerance, max_iter=max_iter)
    eanom_mikkola = _mikkola_solver_wrapper_torch(manom, ecc)
    eanom = torch.where(abs_diff > tolerance, eanom_mikkola, eanom)
    eanom = torch.where(ecc_high, eanom_mikkola, eanom)
    eanom = torch.where(ecc_zero, manom, eanom)
    return eanom

def calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, mass_for_Kamp=None, tau_ref_epoch=58849, tolerance=1e-9, max_iter=10):
    n_orbs = sma.shape[0]
    n_dates = epochs.shape[0]

    if mass_for_Kamp is None:
        mass_for_Kamp = 1.0 * mtot

    ecc_arr = torch.matmul(torch.ones_like(epochs).unsqueeze(-1), ecc.unsqueeze(0))

    period_const = np.sqrt(4*np.pi**2.0*(u.AU)**3/(consts.G*(u.Msun)))
    period_const = period_const.to(u.day).value
    period = torch.sqrt(sma**3/mtot) * period_const
    mean_motion = 2*np.pi/(period)

    manom = (mean_motion*(epochs.unsqueeze(-1) - tau_ref_epoch) - 2*np.pi*tau) % (2.0*np.pi)

    eanom = _calc_ecc_anom_torch(manom, ecc_arr, tolerance=tolerance, max_iter=max_iter)
    tanom = 2.*torch.atan(torch.sqrt((1.0 + ecc)/(1.0 - ecc))*torch.tan(0.5*eanom))
    radius = sma * (1.0 - ecc * torch.cos(eanom))

    c2i2 = torch.cos(0.5*inc)**2
    s2i2 = torch.sin(0.5*inc)**2
    arg1 = tanom + aop + pan
    arg2 = tanom + aop - pan
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    s1 = torch.sin(arg1)
    s2 = torch.sin(arg2)

    raoff = radius * (c2i2*s1 - s2i2*s2) * plx
    deoff = radius * (c2i2*c1 + s2i2*c2) * plx

    Kv_const = np.sqrt(consts.G) * (1.0 * u.Msun) / np.sqrt(1.0 * u.Msun * u.au)
    Kv_const = Kv_const.to(u.km/u.s).value
    Kv = mass_for_Kamp * torch.sqrt(1.0 / ((1.0 - ecc**2) * mtot * sma)) * torch.sin(inc) * Kv_const

    vz = Kv * (ecc*torch.cos(aop) + torch.cos(aop + tanom))
    return raoff, deoff, vz

# --- Params2orbits (Inlined) ---
class Params2orbits(nn.Module):
    def __init__(self, sma_range=[10.0, 1000.0], ecc_range=[0.0, 1.0],
                inc_range=[0.0, np.pi], aop_range=[0.0, 2*np.pi],
                pan_range=[0.0, 2*np.pi], tau_range=[0.0, 1.0],
                plx_range=[56.95-3*0.26, 56.95+3*0.26], mtot_range=[1.22-3*0.08, 1.22+3*0.08]):
        super().__init__()
        self.sma_range = sma_range
        self.ecc_range = ecc_range
        self.inc_range = inc_range
        self.aop_range = aop_range
        self.pan_range = pan_range
        self.tau_range = tau_range
        self.plx_range = plx_range
        self.mtot_range = mtot_range

    def forward(self, params):
        log_sma = np.log(self.sma_range[0]) + params[:, 0] * (np.log(self.sma_range[1])-np.log(self.sma_range[0]))
        sma = torch.exp(log_sma)
        ecc = self.ecc_range[0] + params[:, 1] * (self.ecc_range[1]-self.ecc_range[0])
        inc = torch.acos(np.cos(self.inc_range[1]) + params[:, 2] * (np.cos(self.inc_range[0])-np.cos(self.inc_range[1])))
        aop = self.aop_range[0] + params[:, 3] * (self.aop_range[1]-self.aop_range[0])
        pan = self.pan_range[0] + params[:, 4] * (self.pan_range[1]-self.pan_range[0])
        tau = self.tau_range[0] + params[:, 5] * (self.tau_range[1]-self.tau_range[0])
        plx = self.plx_range[0] + params[:, 6] * (self.plx_range[1]-self.plx_range[0])
        mtot = self.mtot_range[0] + params[:, 7] * (self.mtot_range[1]-self.mtot_range[0])

        return sma, ecc, inc, aop%(2*np.pi), pan%(2*np.pi), tau%1, plx, mtot

# --- Main Script ---
parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for orbit fitting")
parser.add_argument("--divergence_type", default='alpha', type=str, help="KL or alpha, type of objective divergence used for variational inference")
parser.add_argument("--alpha_divergence", default=1.0, type=float, help="hyperparameters for alpha divergence")
parser.add_argument("--save_path", default='./checkpoint/orbit_beta_pic_b/cartesian/alpha1', type=str, help="path to save normalizing flow models")
parser.add_argument("--coordinate_type", default='cartesian', type=str, help="coordinate type")
parser.add_argument("--target", default='betapic', type=str, help="target exoplanet")
parser.add_argument("--data_weight", default=1.0, type=float, help="final data weight for training, between 0-1")
parser.add_argument("--start_order", default=4, type=float, help="start order")
parser.add_argument("--decay_rate", default=3000, type=float, help="decay rate")
parser.add_argument("--n_epoch", default=24000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--n_flow", default=16, type=int, help="number of affine coupling layers in RealNVP")

if __name__ == "__main__":
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu')

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_flow = args.n_flow
    affine = True
    nparams = 8 
    params_generator = RealNVP(nparams, n_flow, affine=affine, seqfrac=1/16, batch_norm=True).to(device)

    target = args.target
    if target == 'betapic':
        # Correct path for sandbox
        astrometry_data = pd.read_csv('./dataset/orbital_fit/betapic_astrometry.csv')
        
        cartesian_indices = np.where(np.logical_not(np.isnan(np.array(astrometry_data['raoff']))))[0]
        polar_indices = np.where(np.logical_not(np.isnan(np.array(astrometry_data['pa']))))[0]
        
        polar_exclude_cartesian_indices = np.where(np.logical_and(np.isnan(np.array(astrometry_data['raoff'])), np.logical_not(np.isnan(np.array(astrometry_data['pa'])))))[0]
        all_indices = np.concatenate([cartesian_indices, polar_exclude_cartesian_indices])

        epochs = torch.tensor(np.array(astrometry_data['epoch']), dtype=torch.float32).to(device)
        sep = torch.tensor(np.array(astrometry_data['sep'][polar_indices]), dtype=torch.float32).to(device)
        sep_err = torch.tensor(np.array(astrometry_data['sep_err'][polar_indices]), dtype=torch.float32).to(device)
        pa_values = np.array(astrometry_data['pa'][polar_indices])
        pa_values[pa_values>180] = pa_values[pa_values>180] - 360
        pa = np.pi / 180 * torch.tensor(pa_values, dtype=torch.float32).to(device)
        pa_err = np.pi / 180 * torch.tensor(np.array(astrometry_data['pa_err'][polar_indices]), dtype=torch.float32).to(device)

        sep_err = sep_err * 3
        pa_err = pa_err * 3

        raoff = torch.tensor(np.array(astrometry_data['raoff'][cartesian_indices]), dtype=torch.float32).to(device)
        raoff_err = torch.tensor(np.array(astrometry_data['raoff_err'][cartesian_indices]), dtype=torch.float32).to(device)
        decoff = torch.tensor(np.array(astrometry_data['decoff'][cartesian_indices]), dtype=torch.float32).to(device)
        decoff_err = torch.tensor(np.array(astrometry_data['decoff_err'][cartesian_indices]), dtype=torch.float32).to(device)

        raoff_convert = sep * torch.sin(pa)
        decoff_convert = sep * torch.cos(pa)

        eps = 1e-3
        orbit_converter = Params2orbits(sma_range=[4, 40], ecc_range=[1e-5, 0.99],
                                        inc_range=[81/180*np.pi, 99/180*np.pi], aop_range=[0.0-eps, 2.0*np.pi+eps],
                                        pan_range=[25/180*np.pi, 85/180*np.pi], tau_range=[0.0-eps, 1.0+eps],
                                        plx_range=[51.44-5*0.12, 51.44+5*0.12], mtot_range=[1.75-5*0.05, 1.75+5*0.05]).to(device)

        coordinate_type = args.coordinate_type
        if coordinate_type == 'cartesian':
            epochs = epochs[cartesian_indices]
        elif coordinate_type == 'polar':
            epochs = epochs[polar_indices]
        elif coordinate_type == 'all' or coordinate_type == 'all_cartesian':
            epochs = epochs[all_indices]

        if coordinate_type == 'cartesian':
            scale_factor = 1.0 / len(cartesian_indices)
        elif coordinate_type == 'polar':
            scale_factor = 1.0 / len(polar_indices) 
        elif coordinate_type == 'all' or coordinate_type == 'all_cartesian':
            scale_factor = 1.0 / len(all_indices)

    else:
        print("Target not supported in this simplified script")
        sys.exit(1)

    n_batch = 64 # Reduced for safety
    lr = 2e-4
    clip = 1e-4
    optimizer = optim.Adam(params_generator.parameters(), lr = lr, amsgrad=True)

    # n_epoch = args.n_epoch
    n_epoch = 10 # Force small for testing
    decay_rate = args.decay_rate
    start_order = args.start_order
    alpha_divergence = args.alpha_divergence
    divergence_type = args.divergence_type
    final_data_weight = args.data_weight

    print("Starting training...")
    for k in range(n_epoch):
        data_weight = min(10**(-start_order+k/decay_rate), final_data_weight)
        z_sample = torch.randn((n_batch, nparams)).to(device=device)

        params_samp, logdet = params_generator.reverse(z_sample)
        params = torch.sigmoid(params_samp)
        det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
        logdet = logdet + det_sigmoid

        sma, ecc, inc, aop, pan, tau, plx, mtot = orbit_converter.forward(params)

        if target == 'betapic':
            raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8, tau_ref_epoch=50000)
        
        sep_torch = torch.transpose(torch.sqrt(raoff_torch**2 + deoff_torch**2), 0, 1)
        pa_torch = torch.transpose(torch.atan2(raoff_torch, deoff_torch), 0, 1)
        raoff_torch = torch.transpose(raoff_torch, 0, 1)
        deoff_torch = torch.transpose(deoff_torch, 0, 1)

        loss_prior = (plx - 51.44)**2 / 0.12**2 + (mtot - 1.75)**2 / 0.05**2
        logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

        if coordinate_type == 'cartesian':
            loss_raoff = (raoff_torch - raoff)**2 / raoff_err**2
            loss_decoff = (deoff_torch - decoff)**2 / decoff_err**2
            loss = data_weight * (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
        else:
             # Simplified for cartesian only as requested
             loss_raoff = (raoff_torch - raoff)**2 / raoff_err**2
             loss_decoff = (deoff_torch - decoff)**2 / decoff_err**2
             loss = data_weight * (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob

        if divergence_type == 'KL' or alpha_divergence == 1:
            loss = torch.mean(scale_factor * loss)
        elif divergence_type == 'alpha':
            rej_weights = nn.Softmax(dim=0)(-(1-alpha_divergence)*loss).detach()
            loss = torch.sum(rej_weights * scale_factor * loss)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params_generator.parameters(), clip)
        optimizer.step()

        print(f"Epoch {k}: Loss {loss.item()}")

    torch.save(params_generator.state_dict(), f"{args.save_path}/final_model.pth")
    print("Training finished.")

```