import torch

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
