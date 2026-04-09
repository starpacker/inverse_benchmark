import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPItorch'))

try:
    import ehtim as eh
except ImportError:
    print("Warning: ehtim not found. Some functionality might fail.")

from generative_model import realnvpfc_model

def torch_complex_matmul(x, F):
    """Complex matrix multiplication for DFT."""
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

def ftmatrix(psize, xdim, ydim, uv):
    """Generate DFT matrix for visibility computation."""
    x = (np.arange(xdim) - xdim / 2) * psize
    y = (np.arange(ydim) - ydim / 2) * psize
    xx, yy = np.meshgrid(x, y)
    coords = np.vstack((xx.flatten(), yy.flatten()))
    phase = -2 * np.pi * (uv @ coords)
    mat = np.exp(1j * phase)
    return mat

class Img_logscale(nn.Module):
    """Learnable log-scale parameter for image intensity."""
    def __init__(self, scale=1):
        super().__init__()
        log_scale = torch.Tensor(np.log(scale) * np.ones(1))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale

def load_and_preprocess_data(obspath, npix, fov, prior_fwhm):
    obs = eh.obsdata.load_uvfits(obspath)
    
    flux_const = np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp'])
    prior_fwhm_rad = prior_fwhm * eh.RADPERUAS
    fov_rad = fov * eh.RADPERUAS
    zbl = flux_const
    
    prior = eh.image.make_square(obs, npix, fov_rad).add_gauss(zbl, (prior_fwhm_rad, prior_fwhm_rad, 0, 0, 0))
    simim = prior.copy()
    simim.ra = obs.ra
    simim.dec = obs.dec
    simim.rf = obs.rf
    
    obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
    uv = np.hstack((obs_data['u'].reshape(-1, 1), obs_data['v'].reshape(-1, 1)))
    
    dft_mat = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv)
    dft_mat = np.expand_dims(dft_mat.T, -1)
    dft_mat = np.concatenate([dft_mat.real, dft_mat.imag], -1)
    dft_mat = torch.tensor(dft_mat, dtype=torch.float32)
    
    # Closure Phase Mapping Logic
    # This section iterates through obs.cphase and maps t1, t2, t3 to visibility indices
    # resulting in cphase_ind_list and cphase_sign_list used by the forward operator.
    
    # Placeholder for closure phase mapping logic
    cphase_ind_list = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]
    cphase_sign_list = [torch.tensor([1]), torch.tensor([-1]), torch.tensor([1])]
    
    data_dict = {
        'dft_mat': dft_mat,
        'cphase_ind_list': cphase_ind_list,
        'cphase_sign_list': cphase_sign_list,
        'obs_data': obs_data
    }
    
    return data_dict

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, npix, device):
    eps = 1e-16
    
    F = dft_mat.to(device=device)
    cphase_ind1 = cphase_ind_list[0].to(device=device)
    cphase_ind2 = cphase_ind_list[1].to(device=device)
    cphase_ind3 = cphase_ind_list[2].to(device=device)
    cphase_sign1 = cphase_sign_list[0].to(device=device)
    cphase_sign2 = cphase_sign_list[1].to(device=device)
    cphase_sign3 = cphase_sign_list[2].to(device=device)
    
    x_flat = torch.reshape(x, (-1, npix * npix)).type(torch.float32).to(device=device)
    vis_torch = torch_complex_matmul(x_flat, F)
    
    vis_amp = torch.sqrt((vis_torch[:, 0, :]) ** 2 + (vis_torch[:, 1, :]) ** 2 + eps)
    
    vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
    vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
    vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)
    
    ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
    ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
    ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
    
    cphase = (cphase_sign1 * ang1 + cphase_sign2 * ang2 + cphase_sign3 * ang3) * 180 / np.pi
    
    return vis_torch, vis_amp, cphase

def run_inversion(data_dict, n_flow, lr, n_epoch, logdet_weight_factor, device):
    npix = data_dict['dft_mat'].shape[0]
    img_generator = realnvpfc_model.RealNVP(n_flow=n_flow, img_size=npix * npix).to(device=device)
    scale_factor = Img_logscale().to(device=device)
    
    optimizer = optim.Adam(list(img_generator.parameters()) + list(scale_factor.parameters()), lr=lr)
    
    for k in range(n_epoch):
        z_sample = torch.randn(32, npix * npix).to(device=device)
        img_samp, logdet = img_generator.reverse(z_sample)
        
        img = torch.nn.Softplus()(img_samp) * scale_factor()
        
        vis_torch, vis_amp, cphase = forward_operator(
            img, data_dict['dft_mat'], data_dict['cphase_ind_list'], data_dict['cphase_sign_list'], npix, device
        )
        
        # Placeholder for loss calculation
        loss_data = torch.tensor(0.0, device=device)
        loss_prior = torch.tensor(0.0, device=device)
        logdet_weight = logdet_weight_factor
        
        loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_weight * torch.mean(logdet)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return img_generator, scale_factor

def evaluate_results(result, data_dict, save_path, device):
    # Placeholder for evaluation logic
    metrics = {}
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radio Interferometry Image Reconstruction')
    parser.add_argument('--obspath', type=str, required=True, help='Path to the .uvfits file')
    parser.add_argument('--npix', type=int, default=128, help='Number of pixels in the image')
    parser.add_argument('--fov', type=float, default=1.0, help='Field of view in microarcseconds')
    parser.add_argument('--prior_fwhm', type=float, default=0.5, help='FWHM of the Gaussian prior in microarcseconds')
    parser.add_argument('--n_flow', type=int, default=8, help='Number of flow steps in the RealNVP model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--logdet_weight_factor', type=float, default=0.1, help='Weight factor for log-determinant loss')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the computations on (cuda or cpu)')
    
    args = parser.parse_args()
    
    data_dict = load_and_preprocess_data(args.obspath, args.npix, args.fov, args.prior_fwhm)
    result = run_inversion(data_dict, args.n_flow, args.lr, args.n_epoch, args.logdet_weight_factor, args.device)
    metrics = evaluate_results(result, data_dict, 'results/', args.device)