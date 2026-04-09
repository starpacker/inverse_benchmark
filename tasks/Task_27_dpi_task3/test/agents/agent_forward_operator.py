import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def torch_complex_matmul(x, F):
    """
    Performs matrix multiplication between a real input x and a complex matrix F.
    
    Args:
        x: Tensor of shape (Batch, N_pixels). The image intensity.
        F: Tensor of shape (N_pixels, N_visibilities, 2). The DFT matrix.
           F[:, :, 0] is the real part, F[:, :, 1] is the imaginary part.
           
    Returns:
        Tensor of shape (Batch, 2, N_visibilities).
        Dimension 1 contains [Real, Imag].
    """
    # Multiply x by the Real part of F
    Fx_real = torch.matmul(x, F[:, :, 0])
    
    # Multiply x by the Imaginary part of F
    Fx_imag = torch.matmul(x, F[:, :, 1])
    
    # Stack them. We unsqueeze to add the complex dimension, then concatenate.
    # Result shape: (Batch, 2, N_visibilities)
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

def forward_operator(x, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device):
    """
    Compute interferometric observables from image x.
    Returns vis_torch, vis_amp, cphase, logcamp.
    """
    eps = 1e-16

    # --- 1. Ensure tensors are on device ---
    # The DFT matrix
    F = dft_mat.to(device=device)
    
    # Indices and signs for Closure Phase (triangles)
    cphase_ind1 = cphase_ind_list[0].to(device=device)
    cphase_ind2 = cphase_ind_list[1].to(device=device)
    cphase_ind3 = cphase_ind_list[2].to(device=device)
    cphase_sign1 = cphase_sign_list[0].to(device=device)
    cphase_sign2 = cphase_sign_list[1].to(device=device)
    cphase_sign3 = cphase_sign_list[2].to(device=device)
    
    # Indices for Closure Amplitude (quadrangles)
    camp_ind1 = camp_ind_list[0].to(device=device)
    camp_ind2 = camp_ind_list[1].to(device=device)
    camp_ind3 = camp_ind_list[2].to(device=device)
    camp_ind4 = camp_ind_list[3].to(device=device)

    # --- 2. Ensure x is on device and properly shaped ---
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device=device)
    # Flatten image: (Batch, H*W)
    x = torch.reshape(x, (-1, npix * npix)).type(torch.float32)

    # --- 3. Compute visibilities via DFT ---
    # Result shape: (Batch, 2, N_visibilities)
    vis_torch = torch_complex_matmul(x, F)

    # --- 4. Visibility amplitude ---
    # sqrt(Real^2 + Imag^2 + epsilon)
    vis_amp = torch.sqrt((vis_torch[:, 0, :]) ** 2 + (vis_torch[:, 1, :]) ** 2 + eps)

    # --- 5. Closure phase ---
    # Select specific visibilities for the triangles
    vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
    vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
    vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)

    # Calculate angles (phases)
    ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
    ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
    ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
    
    # Sum phases and convert to degrees
    cphase = (cphase_sign1 * ang1 + cphase_sign2 * ang2 + cphase_sign3 * ang3) * 180 / np.pi

    # --- 6. Log closure amplitude ---
    # Select specific visibilities for the quadrangles
    vis12_torch = torch.index_select(vis_torch, -1, camp_ind1)
    vis12_amp = torch.sqrt((vis12_torch[:, 0, :]) ** 2 + (vis12_torch[:, 1, :]) ** 2 + eps)
    
    vis34_torch = torch.index_select(vis_torch, -1, camp_ind2)
    vis34_amp = torch.sqrt((vis34_torch[:, 0, :]) ** 2 + (vis34_torch[:, 1, :]) ** 2 + eps)
    
    vis14_torch = torch.index_select(vis_torch, -1, camp_ind3)
    vis14_amp = torch.sqrt((vis14_torch[:, 0, :]) ** 2 + (vis14_torch[:, 1, :]) ** 2 + eps)
    
    vis23_torch = torch.index_select(vis_torch, -1, camp_ind4)
    vis23_amp = torch.sqrt((vis23_torch[:, 0, :]) ** 2 + (vis23_torch[:, 1, :]) ** 2 + eps)

    # Log-ratio calculation
    logcamp = torch.log(vis12_amp) + torch.log(vis34_amp) - torch.log(vis14_amp) - torch.log(vis23_amp)

    return vis_torch, vis_amp, cphase, logcamp

def test_forward_operator():
    # 1. Setup Parameters
    npix = 10
    n_vis = 20  # Number of visibility measurements
    n_cphase = 5 # Number of closure phase triangles
    n_camp = 5   # Number of closure amplitude quadrangles
    batch_size = 2
    device = torch.device("cpu")

    # 2. Create Dummy Data
    # Image: Batch x Height x Width
    x = np.random.rand(batch_size, npix, npix).astype(np.float32)
    
    # DFT Matrix: (Pixels, Visibilities, 2)
    # Flattened pixels = npix * npix
    dft_mat = torch.randn(npix*npix, n_vis, 2)

    # Indices for Closure Phase (must be within range [0, n_vis-1])
    cphase_ind_list = [
        torch.randint(0, n_vis, (n_cphase,)),
        torch.randint(0, n_vis, (n_cphase,)),
        torch.randint(0, n_vis, (n_cphase,))
    ]
    # Signs for Closure Phase (-1 or 1)
    cphase_sign_list = [
        torch.ones(n_cphase),
        torch.ones(n_cphase) * -1,
        torch.ones(n_cphase)
    ]

    # Indices for Closure Amplitude
    camp_ind_list = [
        torch.randint(0, n_vis, (n_camp,)),
        torch.randint(0, n_vis, (n_camp,)),
        torch.randint(0, n_vis, (n_camp,)),
        torch.randint(0, n_vis, (n_camp,))
    ]

    # 3. Run Forward Operator
    vis_torch, vis_amp, cphase, logcamp = forward_operator(
        x, dft_mat, cphase_ind_list, cphase_sign_list, camp_ind_list, npix, device
    )

    # 4. Assertions
    print("Running Unit Test...")
    
    # Check Visibility Shape: (Batch, 2 (Real/Imag), n_vis)
    assert vis_torch.shape == (batch_size, 2, n_vis), f"Vis shape mismatch: {vis_torch.shape}"
    
    # Check Vis Amp Shape: (Batch, n_vis)
    assert vis_amp.shape == (batch_size, n_vis), f"Vis Amp shape mismatch: {vis_amp.shape}"
    
    # Check Closure Phase Shape: (Batch, n_cphase)
    assert cphase.shape == (batch_size, n_cphase), f"CPhase shape mismatch: {cphase.shape}"
    
    # Check Log Closure Amp Shape: (Batch, n_camp)
    assert logcamp.shape == (batch_size, n_camp), f"LogCamp shape mismatch: {logcamp.shape}"

    print("Test Passed: All output shapes are correct.")

if __name__ == "__main__":
    test_forward_operator()