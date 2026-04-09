import os
import tqdm
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Neural Network Components
# ==============================================================================

class G_Renderer(nn.Module):
    def __init__(
        self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False
    ):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act_fn)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats=32, ds_factor=1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            2e-4 * torch.rand((x_mode, y_mode, num_feats)) - 1e-4, requires_grad=True
        )

        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(
            indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False
        )
        self.y0 = nn.Parameter(
            indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False
        )
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def sample(self):
        return (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

    def forward(self):
        return self.sample()


class G_Tensor3D(nn.Module):
    def __init__(
        self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False
    ):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.x0 = None

        self.z_mode = z_dim
        self.z_data = nn.Parameter(
            torch.randn((self.z_mode, self.num_feats)), requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.z_dim = z_dim

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(
            xy[None],
            requires_grad=False,
        )

        if self.x0 is not None:
            device = self.x0.device
            self.x0.data = (indices[:, 0].clamp(min=0, max=x_max - 1)).to(device)
            self.y0.data = indices[:, 1].clamp(min=0, max=y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (xs - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(
                indices[:, 0].clamp(min=0, max=x_max - 1),
                requires_grad=False,
            )
            self.y0 = nn.Parameter(
                indices[:, 1].clamp(min=0, max=y_max - 1),
                requires_grad=False,
            )
            self.x1 = nn.Parameter(
                (self.x0 + 1).clamp(max=x_max - 1), requires_grad=False
            )
            self.y1 = nn.Parameter(
                (self.y0 + 1).clamp(max=y_max - 1), requires_grad=False
            )
            self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        return (self.z_dim - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def sample(self, z):
        z = self.normalize_z(z)
        z0 = z.long().clamp(min=0, max=self.z_dim - 1)
        z1 = (z0 + 1).clamp(max=self.z_dim - 1)
        zlerp_weights = (z - z.long().float())[:, None]

        xy_feat = (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )
        z_feat = (
            self.z_data[z0] * (1.0 - zlerp_weights) + self.z_data[z1] * zlerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)

        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat

        return feat

    def forward(self, z):
        feat = self.sample(z)

        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)

        return out


class FullModel(nn.Module):
    def __init__(
        self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm
    ):
        super().__init__()
        self.img_real = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.img_imag = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.w, self.h = w, h
        self.init_scale_grids(ds_factor=ds_factor)

    def init_scale_grids(self, ds_factor):
        self.img_real.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_real.x_mode,
            y_max=self.img_real.y_mode,
        )
        self.img_imag.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_imag.x_mode,
            y_max=self.img_imag.y_mode,
        )
        self.ds_factor = ds_factor
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode="bilinear")

    def forward(self, dz):
        img_real = self.img_real(dz)
        img_imag = self.img_imag(dz)
        img_real = self.us_module(img_real).squeeze(1)
        img_imag = self.us_module(img_imag).squeeze(1)

        return img_real, img_imag


# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(sample, color, num_modes, device):
    """
    Load FPM data from .mat file and preprocess it.
    Returns all necessary data structures for the inversion.
    """
    if sample == 'Siemens':
        data_struct = sio.loadmat(f"data/{sample}/{sample}_{color}.mat")
        MAGimg = 3
    else:
        raise ValueError("Only Siemens sample is supported in this demo.")
    
    I = data_struct["I_low"].astype("float32")
    I = I[0:int(num_modes), 0:int(num_modes), :]

    M = I.shape[0]
    N = I.shape[1]
    ID_len = I.shape[2]
    
    NAs = data_struct["na_calib"].astype("float32")
    NAx = NAs[:, 0]
    NAy = NAs[:, 1]
    
    if color == "r":
        wavelength = 0.632
    elif color == "g":
        wavelength = 0.5126
    elif color == "b":
        wavelength = 0.471
    else:
        wavelength = 0.632
        
    k0 = 2 * np.pi / wavelength
    mag = data_struct["mag"].astype("float32")
    pixel_size = data_struct["dpix_c"].astype("float32")
    D_pixel = pixel_size / mag
    NA = data_struct["na_cal"].astype("float32")
    kmax = NA * k0
    
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)
    
    Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)
    
    u = -NAx
    v = -NAy
    NAillu = np.sqrt(u**2 + v**2)
    order = np.argsort(NAillu)
    u = u[order]
    v = v[order]
    
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
        ledpos_true[idx, 0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
        ledpos_true[idx, 1] = np.argmin(Fy1_temp)
    
    Isum = I[:, :, order] / np.max(I)
    
    # Angular spectrum - use M, N dimensions (low-res)
    kxx, kyy = np.meshgrid(
        np.linspace(-np.pi/D_pixel, np.pi/D_pixel, M),
        np.linspace(-np.pi/D_pixel, np.pi/D_pixel, N)
    )
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)
    
    # Pupil support
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1
    
    Pupil0 = torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)
    
    data_dict = {
        'Isum': Isum,
        'Pupil0': Pupil0,
        'kzz': kzz,
        'ledpos_true': ledpos_true,
        'M': M,
        'N': N,
        'MM': MM,
        'NN': NN,
        'ID_len': ID_len,
        'MAGimg': MAGimg,
    }
    
    return data_dict


# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_operator(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, mag):
    """
    Forward model: compute sub-spectrum intensities from complex image.
    This implements the FPM forward model:
    1. FFT of complex image
    2. Pad to high-res spectrum
    3. Extract sub-apertures for each LED
    4. Apply pupil mask and defocus
    5. IFFT and compute intensity
    """
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    
    to_pad_x = (spectrum_mask.shape[-2] * mag - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * mag - O.shape[-1]) // 2
    O = F.pad(O, (to_pad_y, to_pad_y, to_pad_x, to_pad_x, 0, 0), "constant", 0)

    O_sub = torch.stack(
        [O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))], dim=1
    )
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub)

    return oI_sub


# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(data_dict, num_epochs, num_feats, num_modes, use_layernorm, use_amp, device, vis_dir, is_os):
    """
    Run the FPM-INR optimization to reconstruct the high-resolution complex image.
    """
    Isum = data_dict['Isum']
    Pupil0 = data_dict['Pupil0']
    kzz = data_dict['kzz']
    ledpos_true = data_dict['ledpos_true']
    M = data_dict['M']
    N = data_dict['N']
    MM = data_dict['MM']
    ID_len = data_dict['ID_len']
    MAGimg = data_dict['MAGimg']
    
    z_min = 0.0
    z_max = 1.0
    led_batch_size = 1
    cur_ds = 1
    lr_decay_step = 6
    
    model = FullModel(
        w=MM,
        h=MM,
        num_feats=num_feats,
        x_mode=num_modes,
        y_mode=num_modes,
        z_min=z_min,
        z_max=z_max,
        ds_factor=cur_ds,
        use_layernorm=use_layernorm,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        lr=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters()),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.1
    )
    
    model_fn = model
    
    t = tqdm.trange(num_epochs)
    final_amplitude = None
    final_phase = None
    final_loss = None
    final_psnr = None
    
    for epoch in t:
        led_indices = list(np.arange(ID_len))
        dzs = torch.FloatTensor([0.0]).to(device)
        
        if epoch == 0:
            if is_os == "Windows":
                try:
                    model_fn = torch.jit.trace(model, dzs[0:1])
                except Exception:
                    model_fn = model
            elif is_os == "Linux":
                try:
                    model_fn = torch.compile(model, backend="inductor")
                except Exception:
                    model_fn = model
            else:
                model_fn = model

        for dz in dzs:
            dz = dz.unsqueeze(0)
            
            for it in range(ID_len // led_batch_size):
                model.zero_grad()
                
                # Compute defocus mask at low-res size
                dfmask = torch.exp(
                    1j * kzz.repeat(dz.shape[0], 1, 1)
                    * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
                )
                
                led_num = led_indices[it * led_batch_size : (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                
                # spectrum_mask is at low-res size (M, N)
                spectrum_mask_ampli = Pupil0.repeat(len(dz), len(led_num), 1, 1) * torch.abs(dfmask)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * torch.angle(dfmask)
                spectrum_mask = spectrum_mask_ampli * torch.exp(1j * spectrum_mask_phase)
                
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    img_ampli, img_phase = model_fn(dz)
                    img_complex = img_ampli * torch.exp(1j * img_phase)
                    
                    uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                    x_0 = vo - M // 2
                    x_1 = vo + M // 2
                    y_0 = uo - N // 2
                    y_1 = uo + N // 2
                    
                    oI_cap = torch.sqrt(Isum[:, :, led_num])
                    oI_cap = oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)
                    
                    oI_sub = forward_operator(
                        img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                    )
                    
                    loss = F.smooth_l1_loss(oI_cap, oI_sub)
                    mse_loss = F.mse_loss(oI_cap, oI_sub)
                    
                loss.backward()
                
                psnr_val = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr_val:.2f}")
                optimizer.step()
                
                final_loss = loss.item()
                final_psnr = psnr_val
                
        scheduler.step()
        
        # Store final results
        final_amplitude = img_ampli[0].float().cpu().detach().numpy()
        final_phase = img_phase[0].float().cpu().detach().numpy()
        
        # Visualization
        if (epoch + 1) % 1 == 0:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            im = axs[0].imshow(final_amplitude, cmap="gray")
            axs[0].axis("image")
            axs[0].set_title("Reconstructed amplitude")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = axs[1].imshow(final_phase, cmap="gray")
            axs[1].axis("image")
            axs[1].set_title("Reconstructed phase")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            plt.savefig(f"{vis_dir}/epoch_{epoch}.png")
            plt.close()
    
    result = {
        'amplitude': final_amplitude,
        'phase': final_phase,
        'model': model,
        'final_loss': final_loss,
        'final_psnr': final_psnr,
    }
    
    return result


# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(result, vis_dir, sample, color):
    """
    Evaluate and save the reconstruction results.
    """
    amplitude = result['amplitude']
    phase = result['phase']
    model = result['model']
    final_loss = result['final_loss']
    final_psnr = result['final_psnr']
    
    # Compute statistics
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_min = np.min(amplitude)
    amp_max = np.max(amplitude)
    
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    
    metrics = {
        'final_loss': final_loss,
        'final_psnr': final_psnr,
        'amplitude_mean': amp_mean,
        'amplitude_std': amp_std,
        'amplitude_min': amp_min,
        'amplitude_max': amp_max,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'phase_min': phase_min,
        'phase_max': phase_max,
    }
    
    print("\n=== Reconstruction Results ===")
    print(f"Final Loss: {final_loss:.6e}")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Amplitude - Mean: {amp_mean:.4f}, Std: {amp_std:.4f}, Range: [{amp_min:.4f}, {amp_max:.4f}]")
    print(f"Phase - Mean: {phase_mean:.4f}, Std: {phase_std:.4f}, Range: [{phase_min:.4f}, {phase_max:.4f}]")
    
    # Save model
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{sample}_{color}.pth')
    
    tensors_to_save = []
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    torch.save(tensors_to_save, save_path)
    print(f"Model saved to {save_path}")
    
    # Save final images
    np.save(os.path.join(vis_dir, 'final_amplitude.npy'), amplitude)
    np.save(os.path.join(vis_dir, 'final_phase.npy'), phase)
    print(f"Results saved to {vis_dir}")
    
    return metrics


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Settings
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    num_epochs = 5
    num_feats = 32
    num_modes = 512
    use_layernorm = False
    use_amp = True
    sample = "Siemens"
    color = "r"
    is_os = "Linux"
    
    print(f"Running FPM-INR for {sample} ({color}) with {num_epochs} epochs...")
    
    vis_dir = f"./vis/feat{num_feats}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(sample, color, num_modes, device)
    
    # Step 2 is embedded in run_inversion (forward_operator is called inside)
    
    # Step 3: Run inversion
    print("Step 3: Running inversion...")
    result = run_inversion(
        data_dict, num_epochs, num_feats, num_modes, 
        use_layernorm, use_amp, device, vis_dir, is_os
    )
    
    # Step 4: Evaluate results
    print("Step 4: Evaluating results...")
    metrics = evaluate_results(result, vis_dir, sample, color)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")