import torch
import torch.nn as nn
import numpy as np

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

def torch_complex_matmul(x, F):
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device):
    eps = 1e-16
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

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device=device)
    x = torch.reshape(x, (-1, npix * npix)).type(torch.float32)

    vis_torch = torch_complex_matmul(x, F)
    vis_amp = torch.sqrt((vis_torch[:, 0, :]) ** 2 + (vis_torch[:, 1, :]) ** 2 + eps)

    vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
    vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
    vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)

    ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
    ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
    ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
    cphase = (cphase_sign1 * ang1 + cphase_sign2 * ang2 + cphase_sign3 * ang3) * 180 / np.pi

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

def evaluate_results(result, preprocessed_data, n_samples=100, n_gaussian=2, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npix = preprocessed_data['npix']
    fov = preprocessed_data['fov']
    flux_const = preprocessed_data['flux_const']

    nparams = result['nparams']
    n_flow = result['n_flow']

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

    with torch.no_grad():
        z_sample = torch.randn((n_samples, nparams)).to(device=device)
        params_samp, _ = params_generator.reverse(z_sample)
        params = torch.sigmoid(params_samp)
        imgs = img_converter.forward(params)

    imgs_np = imgs.cpu().numpy()

    mean_img = np.mean(imgs_np, axis=0)
    std_img = np.std(imgs_np, axis=0)
    total_flux_samples = np.sum(imgs_np, axis=(1, 2))

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