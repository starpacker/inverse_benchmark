import torch

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
