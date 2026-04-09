import numpy as np

import torch

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data

import matplotlib.pyplot as plt

KEY_FINGERPRINTS = 'fingerprints'

KEY_MR_PARAMS = 'mr_params'

ID_MAP_FF = 'FFmap'

ID_MAP_T1H2O = 'T1H2Omap'

ID_MAP_T1FAT = 'T1FATmap'

ID_MAP_B0 = 'B0map'

ID_MAP_B1 = 'B1map'

MR_PARAMS = (ID_MAP_FF, ID_MAP_T1H2O, ID_MAP_T1FAT, ID_MAP_B0, ID_MAP_B1)

def de_normalize(data: np.ndarray, minmax_tuple: tuple):
    return data * (minmax_tuple[1] - minmax_tuple[0]) + minmax_tuple[0]

def de_normalize_mr_parameters(data: np.ndarray, mr_param_ranges, mr_params=MR_PARAMS):
    data_de_normalized = data.copy()
    for idx, mr_param in enumerate(mr_params):
        if mr_param in mr_param_ranges:
             data_de_normalized[:, idx] = de_normalize(data[:, idx], mr_param_ranges[mr_param])
    return data_de_normalized

def forward_operator(model, x, ndim_y, device):
    """
    Simulates the forward Bloch process (learning-based).
    Maps parameters x -> fingerprint y.
    
    Because the INN has fixed input/output dimension equal to ndim_y,
    and x usually has fewer dimensions (ndim_x), we pad x with zeros.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    
    x = x.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
        
    current_bs = x.size(0)
    ndim_x = x.size(1)
    
    pad_len = ndim_y - ndim_x
    if pad_len < 0:
        raise ValueError("Parameter dimension cannot be larger than fingerprint dimension for this INN architecture.")
        
    pad_x = torch.zeros(current_bs, pad_len, device=device)
    x_padded = torch.cat((x, pad_x), dim=1)
    
    # Forward pass through INN (x -> y)
    y_pred = model(x_padded, rev=False)
    
    return y_pred

def run_inversion(model, y, ndim_x, device):
    """
    Performs the inversion (Reconstruction).
    Maps fingerprint y -> parameters x.
    """
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()
        
    y = y.to(device)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    
    # Inverse pass through INN (y -> x_padded)
    x_rec_padded = model(y, rev=True)
    
    # Crop the padding to get the actual parameters
    x_rec = x_rec_padded[:, :ndim_x]
    
    return x_rec.detach()

def evaluate_results(model, dataloader, dataset, dims, device, epochs):
    """
    Trains the INN model and then evaluates it on a sample.
    """
    ndim_x = dims['ndim_x']
    ndim_y = dims['ndim_y']
    
    # -----------------------
    # TRAINING LOOP
    # -----------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[KEY_MR_PARAMS].to(device)
            y = batch[KEY_FINGERPRINTS].to(device)
            current_bs = x.size(0)
            
            # Pad x to match y dimension for INN
            pad_x = torch.zeros(current_bs, ndim_y - ndim_x, device=device)
            x_padded = torch.cat((x, pad_x), dim=1)
            
            optimizer.zero_grad()
            
            # Forward loss: predict y from x
            y_hat = model(x_padded, rev=False)
            loss_fwd = F.mse_loss(y_hat, y)
            
            # Backward loss: predict x from y
            x_hat_padded = model(y, rev=True)
            loss_bwd = F.mse_loss(x_hat_padded, x_padded)
            
            loss = loss_fwd + loss_bwd
            loss.backward()
            
            # Gradient Clipping
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-15.00, 15.00)
            
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    # -----------------------
    # EVALUATION
    # -----------------------
    model.eval()
    
    # Get a single sample for evaluation
    sample_idx = 0
    sample = dataset[sample_idx]
    x_gt_np = sample[KEY_MR_PARAMS]
    y_gt_np = sample[KEY_FINGERPRINTS]
    
    # 1. Inversion (y -> x)
    x_rec = run_inversion(model, y_gt_np, ndim_x, device)
    x_rec_np = x_rec.cpu().numpy()
    
    # 2. Forward (x -> y)
    y_pred = forward_operator(model, x_gt_np, ndim_y, device)
    y_pred_np = y_pred.detach().cpu().numpy()

    # 3. Denormalize parameters for display
    x_gt_denorm = de_normalize_mr_parameters(x_gt_np[np.newaxis, :], dataset.mr_param_ranges)
    x_rec_denorm = de_normalize_mr_parameters(x_rec_np, dataset.mr_param_ranges)
    
    print("\nReconstruction Results (Parameters):")
    param_names = MR_PARAMS
    for i in range(ndim_x):
        name = param_names[i] if i < len(param_names) else f"Param {i}"
        err = abs(x_gt_denorm[0,i] - x_rec_denorm[0,i])
        print(f"{name}: GT = {x_gt_denorm[0,i]:.4f}, Pred = {x_rec_denorm[0,i]:.4f}, Error = {err:.4f}")
        
    mse_fp = np.mean((y_pred_np - y_gt_np)**2)
    print(f"\nForward Model Fingerprint MSE: {mse_fp:.6f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_gt_np, label='Ground Truth')
    plt.plot(y_pred_np[0], label='Predicted (Learned Bloch)')
    plt.title('MR Fingerprint Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('mrf_fingerprint_comparison.png')
    print("Saved fingerprint comparison plot to mrf_fingerprint_comparison.png")
    
    return x_rec_denorm
