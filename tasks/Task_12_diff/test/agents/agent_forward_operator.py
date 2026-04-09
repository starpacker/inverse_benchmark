import torch
import diffmetrology as dm

def forward_operator(DM, angle, valid_cap, mode='trace'):
    """
    Forward operator that performs ray tracing or rendering.
    
    Parameters:
    -----------
    DM : DiffMetrology object
        The metrology object containing scene and lensgroup
    angle : float
        Rotation angle for the measurement
    valid_cap : torch.Tensor
        Valid pixel mask
    mode : str
        'trace' for ray tracing (returns intersection points)
        'render' for image rendering (returns rendered images)
    
    Returns:
    --------
    result : torch.Tensor
        Either intersection points (mode='trace') or rendered images (mode='render')
    """
    if mode == 'trace':
        # Ray tracing forward model - computes intersection points on display
        # DM.trace returns a tuple/list; we stack the first element.
        # We slice [..., 0:2] to keep only x and y coordinates.
        ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
        return ps
    elif mode == 'render':
        # Image rendering forward model - renders images from current parameters
        # The rendered output is masked by valid_cap immediately.
        I = valid_cap * torch.stack(DM.render(with_element=True, angles=angle))
        
        # Numerical stability: Replace NaNs with 0.0
        I[torch.isnan(I)] = 0.0
        return I
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trace' or 'render'.")