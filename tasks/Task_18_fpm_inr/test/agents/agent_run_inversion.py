import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
import os

class G_Renderer(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False):
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
        return self.net(x)

class G_Tensor3D(nn.Module):
    def __init__(self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.z_dim = z_dim
        self.z_data = nn.Parameter(torch.randn((self.z_dim, self.num_feats)))
        self.z_min, self.z_max = z_min, z_max
        self.x0 = None

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xy.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        
        self.x0 = nn.Parameter(indices[:, 0].clamp(min=0, max=x_max - 1), requires_grad=False)
        self.y0 = nn.Parameter(indices[:, 1].clamp(min=0, max=y_max - 1), requires_grad=False)
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_max - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_max - 1), requires_grad=False)
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def sample(self, z):
        if self.x0 is None:
            raise RuntimeError("Coordinates not initialized! Call create_coords first.")
            
        Ia = self.data[self.y0, self.x0]
        Ib = self.data[self.y0, self.x1]
        Ic = self.data[self.y1, self.x0]
        Id = self.data[self.y1, self.x1]
        
        wa = (1.0 - self.lerp_weights[:, 0:1]) * (1.0 - self.lerp_weights[:, 1:2])
        wb = self.lerp_weights[:, 0:1] * (1.0 - self.lerp_weights[:, 1:2])
        wc = (1.0 - self.lerp_weights[:, 0:1]) * self.lerp_weights[:, 1:2]
        wd = self.lerp_weights[:, 0:1] * self.lerp_weights[:, 1:2]
        
        xy_feat = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return xy_feat

    def forward(self, z):
        feat = self.sample(z)
        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)
        return out

class FullModel(nn.Module):
    def __init__(self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm):
        super().__init__()
        self.img_real = G_Tensor3D(x_mode, y_mode, 1, z_min, z_max, num_feats, use_layernorm)
        self.img_imag = G_Tensor3D(x_mode, y_mode, 1, z_min, z_max, num_feats, use_layernorm)
        self.w, self.h = w, h
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode='bilinear')
        
        self.init_scale_grids(ds_factor)

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

    def forward(self, z):
        real = self.img_real(z)
        imag = self.img_imag(z)
        
        real = self.us_module(real).squeeze(1)
        imag = self.us_module(imag).squeeze(1)
        
        return real, imag

def forward_operator(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg):
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    
    to_pad_x = (spectrum_mask.shape[-2] * MAGimg - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * MAGimg - O.shape[-1]) // 2
    O = F.pad(O, (to_pad_y, to_pad_y, to_pad_x, to_pad_x, 0, 0), "constant", 0)

    O_sub = torch.stack(
        [O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))], dim=1
    )
    
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub) 

    return oI_sub

def run_inversion(data_dict, num_epochs, num_feats, num_modes, use_layernorm, use_amp, device, vis_dir, is_os):
    Isum = torch.as_tensor(data_dict['Isum']).float().to(device)
    Pupil0 = torch.as_tensor(data_dict['Pupil0']).type(torch.complex64).to(device)
    kzz = torch.as_tensor(data_dict['kzz']).float().to(device)
    ledpos_true = torch.as_tensor(data_dict['ledpos_true']).long().to(device)
    
    M = data_dict['M']
    N = data_dict['N']
    MM = data_dict['MM']
    ID_len = data_dict['ID_len']
    MAGimg = data_dict['MAGimg']

    model = FullModel(
        w=MM, h=MM, num_feats=num_feats, x_mode=num_modes, y_mode=num_modes,
        z_min=0.0, z_max=1.0, ds_factor=1, use_layernorm=use_layernorm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    final_loss = 0.0
    final_psnr = 0.0
    
    t = tqdm.trange(num_epochs)
    
    for epoch in t:
        dzs = torch.FloatTensor([0.0]).to(device)
        
        for dz in dzs:
            dz = dz.unsqueeze(0)
            
            dfmask = torch.exp(
                1j * kzz.repeat(dz.shape[0], 1, 1)
                * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
            )
            
            led_batch_size = 1
            for it in range(ID_len // led_batch_size):
                optimizer.zero_grad()
                
                led_num = list(range(it * led_batch_size, (it + 1) * led_batch_size))
                
                current_dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                spectrum_mask = Pupil0.repeat(len(dz), len(led_num), 1, 1) * current_dfmask

                img_real, img_imag = model(dz)
                img_complex = img_real * torch.exp(1j * img_imag)
                
                uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                x_0 = vo - M // 2
                x_1 = vo + M // 2
                y_0 = uo - N // 2
                y_1 = uo + N // 2
                
                oI_sub = forward_operator(
                    img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                )
                
                oI_cap = torch.sqrt(Isum[:, :, led_num]).permute(2, 0, 1).unsqueeze(0)
                
                loss = F.smooth_l1_loss(oI_cap, oI_sub)
                mse_loss = F.mse_loss(oI_cap, oI_sub)
                
                loss.backward()
                optimizer.step()
                
                final_loss = loss.item()
                final_psnr = 10 * -torch.log10(mse_loss).item()
                
                t.set_postfix(Loss=f"{final_loss:.4e}", PSNR=f"{final_psnr:.2f}")

    return {
        'amplitude': img_real[0].detach().cpu().numpy(), 
        'phase': img_imag[0].detach().cpu().numpy(),
        'model': model,
        'final_loss': final_loss,
        'final_psnr': final_psnr,
    }