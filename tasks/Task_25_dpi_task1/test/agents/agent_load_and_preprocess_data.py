import torch
import torch.nn as nn
import numpy as np

def torch_complex_matmul(x, F):
    """Complex matrix multiplication for DFT.
    x: (batch, pixels) - Real valued image
    F: (pixels, visibilities, 2) - Complex DFT matrix split into Real/Imag
    """
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
    """
    Load observation data and create prior image.
    Correctly handles ehtim add_gauss syntax and ensures imports are valid.
    """
    import ehtim as eh

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
    
    obs.add_cphase(count='min-cut0bl', uv_min=.1e9)
    obs.add_camp(debias=True, count='min')
    obs.add_logcamp(debias=True, count='min')
    
    zero_symbol = 100000
    n_cphase = len(obs.cphase['time'])
    cphase_map = np.zeros((n_cphase, 3))
    
    for k1 in range(n_cphase):
        t_match = np.where(obs.data['time'] == obs.cphase['time'][k1])[0]
        for k2 in t_match:
            t1_data = obs.data['t1'][k2]
            t2_data = obs.data['t2'][k2]
            cp_t1 = obs.cphase['t1'][k1]
            cp_t2 = obs.cphase['t2'][k1]
            cp_t3 = obs.cphase['t3'][k1]
            
            if t1_data == cp_t1 and t2_data == cp_t2:
                cphase_map[k1, 0] = zero_symbol if k2 == 0 else k2
            elif t2_data == cp_t1 and t1_data == cp_t2:
                cphase_map[k1, 0] = -zero_symbol if k2 == 0 else -k2
            
            if t1_data == cp_t2 and t2_data == cp_t3:
                cphase_map[k1, 1] = zero_symbol if k2 == 0 else k2
            elif t2_data == cp_t2 and t1_data == cp_t3:
                cphase_map[k1, 1] = -zero_symbol if k2 == 0 else -k2
            
            if t1_data == cp_t3 and t2_data == cp_t1:
                cphase_map[k1, 2] = zero_symbol if k2 == 0 else k2
            elif t2_data == cp_t3 and t1_data == cp_t1:
                cphase_map[k1, 2] = -zero_symbol if k2 == 0 else -k2
    
    cphase_ind1 = np.abs(cphase_map[:, 0]).astype(int)
    cphase_ind1[cphase_ind1 == zero_symbol] = 0
    cphase_ind2 = np.abs(cphase_map[:, 1]).astype(int)
    cphase_ind2[cphase_ind2 == zero_symbol] = 0
    cphase_ind3 = np.abs(cphase_map[:, 2]).astype(int)
    cphase_ind3[cphase_ind3 == zero_symbol] = 0
    
    cphase_sign1 = np.sign(cphase_map[:, 0])
    cphase_sign2 = np.sign(cphase_map[:, 1])
    cphase_sign3 = np.sign(cphase_map[:, 2])
    
    cphase_ind_list = [
        torch.tensor(cphase_ind1, dtype=torch.long),
        torch.tensor(cphase_ind2, dtype=torch.long),
        torch.tensor(cphase_ind3, dtype=torch.long)
    ]
    cphase_sign_list = [
        torch.tensor(cphase_sign1, dtype=torch.float32),
        torch.tensor(cphase_sign2, dtype=torch.float32),
        torch.tensor(cphase_sign3, dtype=torch.float32)
    ]
    
    vis_true = np.concatenate([
        np.expand_dims(obs.data['vis'].real, 0),
        np.expand_dims(obs.data['vis'].imag, 0)
    ], 0)
    visamp_true = np.abs(obs.data['vis'])
    cphase_true = np.array(obs.cphase['cphase'])
    prior_im = np.array(prior.imvec.reshape((npix, npix)))
    
    sigma_vis = obs.data['sigma']
    sigma_cphase = obs.cphase['sigmacp']
    
    n_camp = len(obs.camp['camp']) if len(obs.camp['camp']) > 0 else 1
    
    data_dict = {
        'obs': obs,
        'dft_mat': dft_mat,
        'cphase_ind_list': cphase_ind_list,
        'cphase_sign_list': cphase_sign_list,
        'vis_true': vis_true,
        'visamp_true': visamp_true,
        'cphase_true': cphase_true,
        'prior_im': prior_im,
        'sigma_vis': sigma_vis,
        'sigma_cphase': sigma_cphase,
        'flux_const': flux_const,
        'npix': npix,
        'n_camp': n_camp
    }
    
    return data_dict

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, npix, device):
    """
    Compute forward model: image -> visibilities -> closure phases.
    Uses vectorized operations for efficiency.
    """
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
    """Run the inversion optimization using normalizing flow."""
    npix = data_dict['npix']
    flux_const = data_dict['flux_const']
    dft_mat = data_dict['dft_mat']
    cphase_ind_list = data_dict['cphase_ind_list']
    cphase_sign_list = data_dict['cphase_sign_list']
    
    cphase_true = torch.tensor(data_dict['cphase_true'], dtype=torch.float32).to(device=device)
    prior_im = torch.tensor(data_dict['prior_im'], dtype=torch.float32).to(device=device)
    sigma_cphase = torch.tensor(data_dict['sigma_cphase'], dtype=torch.float32).to(device=device)
    n_camp = data_dict['n_camp']
    
    img_generator = realnvpfc_model.RealNVP(npix * npix, n_flow, affine=True).to(device)
    logscale_factor = Img_logscale(scale=flux_const / (0.8 * npix * npix)).to(device)
    
    optimizer = optim.Adam(
        list(img_generator.parameters()) + list(logscale_factor.parameters()),
        lr=lr
    )
    
    cphase_weight = len(data_dict['cphase_true']) / n_camp
    imgflux_weight = 1000.0
    imgcrossentropy_weight = 1024.0
    logdet_weight = 2.0 * logdet_weight_factor / n_camp
    
    loss_history = []
    
    for k in range(n_epoch):
        z_sample = torch.randn(32, npix * npix).to(device=device)
        img_samp, logdet = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, npix, npix))
        
        logscale_factor_value = logscale_factor.forward()
        scale_factor = torch.exp(logscale_factor_value)
        img = torch.nn.Softplus()(img_samp) * scale_factor
        
        det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
        det_scale = logscale_factor_value * npix * npix
        logdet = logdet + det_softplus + det_scale
        
        vis_torch, vis_amp, cphase = forward_operator(
            img, dft_mat, cphase_ind_list, cphase_sign_list, npix, device
        )
        
        loss_cross_entropy = torch.mean(
            img * (torch.log(img + 1e-12) - torch.log(prior_im + 1e-12)),
            (-1, -2)
        )
        
        img_flux = torch.sum(img, (-1, -2))
        loss_flux = (img_flux - flux_const) ** 2
        
        angle_true = cphase_true * np.pi / 180
        angle_pred = cphase * np.pi / 180
        loss_cphase = 2.0 * torch.mean(
            (1 - torch.cos(angle_true - angle_pred)) / (sigma_cphase * np.pi / 180) ** 2,
            1
        )
        
        loss_data = cphase_weight * loss_cphase
        loss_prior = imgcrossentropy_weight * loss_cross_entropy + imgflux_weight * loss_flux
        
        loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_weight * torch.mean(logdet)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(img_generator.parameters()) + list(logscale_factor.parameters()),
            0.1
        )
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if k % 10 == 0:
            print(f"Epoch {k}: Loss {loss.item():.4f}, CPhase {torch.mean(loss_cphase).item():.4f}")
    
    with torch.no_grad():
        z_sample = torch.randn(32, npix * npix).to(device=device)
        img_samp, _ = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, npix, npix))
        logscale_factor_value = logscale_factor.forward()
        scale_factor = torch.exp(logscale_factor_value)
        final_images = torch.nn.Softplus()(img_samp) * scale_factor
    
    result = {
        'model': img_generator,
        'logscale_factor': logscale_factor,
        'final_images': final_images.detach().cpu().numpy(),
        'loss_history': np.array(loss_history),
        'npix': npix
    }
    
    return result

def evaluate_results(result, data_dict, save_path, device):
    """Evaluate and save the inversion results."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    final_images = result['final_images']
    loss_history = result['loss_history']
    npix = result['npix']
    
    mean_image = np.mean(final_images, axis=0)
    std_image = np.std(final_images, axis=0)
    
    total_flux = np.sum(mean_image)
    target_flux = data_dict['flux_const']
    flux_error = np.abs(total_flux - target_flux) / target_flux * 100
    
    final_loss = loss_history[-1] if len(loss_history) > 0 else float('nan')
    
    dft_mat = data_dict['dft_mat']
    cphase_ind_list = data_dict['cphase_ind_list']
    cphase_sign_list = data_dict['cphase_sign_list']
    cphase_true = data_dict['cphase_true']
    
    mean_image_tensor = torch.tensor(mean_image, dtype=torch.float32).unsqueeze(0).to(device)
    
    _, _, cphase_pred = forward_operator(
        mean_image_tensor, dft_mat, cphase_ind_list, cphase_sign_list, npix, device
    )
    cphase_pred_np = cphase_pred.detach().cpu().numpy().flatten()
    
    cphase_residual = np.abs(cphase_true - cphase_pred_np)
    cphase_residual = np.minimum(cphase_residual, 360 - cphase_residual)
    mean_cphase_error = np.mean(cphase_residual)
    
    torch.save(result['model'].state_dict(), os.path.join(save_path, 'model.pth'))
    np.save(os.path.join(save_path, 'reconstruction.npy'), final_images)
    np.save(os.path.join(save_path, 'mean_image.npy'), mean_image)
    np.save(os.path.join(save_path, 'std_image.npy'), std_image)
    np.save(os.path.join(save_path, 'loss_history.npy'), loss_history)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Mean image shape: {mean_image.shape}")
    print(f"Total flux: {total_flux:.4f} (target: {target_flux:.4f})")
    print(f"Flux error: {flux_error:.2f}%")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Mean closure phase error: {mean_cphase_error:.2f} degrees")
    print(f"Results saved to: {save_path}")
    
    metrics = {
        'mean_image': mean_image,
        'std_image': std_image,
        'total_flux': total_flux,
        'target_flux': target_flux,
        'flux_error_percent': flux_error,
        'final_loss': final_loss,
        'mean_cphase_error_deg': mean_cphase_error
    }
    
    return metrics