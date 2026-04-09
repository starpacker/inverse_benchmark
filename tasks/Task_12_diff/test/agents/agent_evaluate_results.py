import numpy as np
import torch
import diffmetrology as dm
import matplotlib.pyplot as plt

def forward_operator(DM, angle, valid_cap, mode='trace'):
    """
    Forward operator that performs ray tracing or rendering.
    
    Parameters:
    -----------
    DM : DiffMetrology object
        The metrology object containing scene and lensgroup.
    angle : float
        Rotation angle for the measurement.
    valid_cap : torch.Tensor
        Boolean mask indicating valid pixels/rays.
    mode : str
        'trace' for ray tracing (returns intersection points).
        'render' for image rendering (returns rendered images).
    
    Returns:
    --------
    result : torch.Tensor
        Either intersection points (N, ..., 2) or rendered images (N, H, W).
    """
    if mode == 'trace':
        ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
        return ps
    elif mode == 'render':
        I = valid_cap * torch.stack(DM.render(with_element=True, angles=angle))
        I[torch.isnan(I)] = 0.0
        return I
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trace' or 'render'.")

def evaluate_results(DM, ps_cap, valid_cap, angle, I0, save_string="result"):
    """
    Evaluate and visualize the results of the inversion.
    
    Parameters:
    -----------
    DM : DiffMetrology object
        The metrology object with current parameters.
    ps_cap : torch.Tensor
        Captured/measured intersection points (Ground Truth).
    valid_cap : torch.Tensor
        Valid pixel mask.
    angle : float
        Rotation angle.
    I0 : torch.Tensor
        Reference images (Ground Truth) for comparison.
    save_string : str
        Prefix for saved figure filenames.
    
    Returns:
    --------
    error : float
        Mean displacement error in mm.
    """
    print("Showing lens parameters:")
    for i in range(len(DM.scene.lensgroup.surfaces)):
        print(f"Lens radius of curvature at surface[{i}]: {1.0 / DM.scene.lensgroup.surfaces[i].c.item()}")
    print(DM.scene.lensgroup.surfaces[1].d)

    print("Visualize status.")
    
    ps_current = forward_operator(DM, angle, valid_cap, mode='trace')

    print("Showing spot diagrams at display.")
    DM.spot_diagram(ps_cap, ps_current, valid=valid_cap, angle=angle, with_grid=False)
    plt.savefig(f"{save_string}_spot_diagram.png")
    plt.close()

    print("Showing images (measurement & modeled & |measurement - modeled|).")

    I = forward_operator(DM, angle, valid_cap, mode='render')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        im = axes[i, 0].imshow(I0[i].cpu(), vmin=0, vmax=1, cmap='gray')
        axes[i, 0].set_title(f"Camera {i + 1}\nMeasurement")
        axes[i, 0].set_xlabel('[pixel]')
        axes[i, 0].set_ylabel('[pixel]')
        plt.colorbar(im, ax=axes[i, 0])

        im = axes[i, 1].imshow(I[i].cpu().detach(), vmin=0, vmax=1, cmap='gray')
        plt.colorbar(im, ax=axes[i, 1])
        axes[i, 1].set_title(f"Camera {i + 1}\nModeled")
        axes[i, 1].set_xlabel('[pixel]')
        axes[i, 1].set_ylabel('[pixel]')

        im = axes[i, 2].imshow(I0[i].cpu() - I[i].cpu().detach(), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im, ax=axes[i, 2])
        axes[i, 2].set_title(f"Camera {i + 1}\nError")
        axes[i, 2].set_xlabel('[pixel]')
        axes[i, 2].set_ylabel('[pixel]')

    fig.suptitle(save_string)
    fig.savefig(f"{save_string}_images.jpg", bbox_inches='tight')
    plt.close()

    T = ps_current - ps_cap
    E = torch.sqrt(torch.sum(T[valid_cap, ...] ** 2, axis=-1)).mean()
    
    print("error = {} [um]".format(E * 1e3))

    return E.item()