import numpy as np
import torch
import diffmetrology as dm
from matplotlib.image import imread
import matplotlib.pyplot as plt

def forward_operator(DM, angle, valid_cap, mode='trace'):
    """
    Forward operator that performs ray tracing or rendering.
    """
    if mode == 'trace':
        # Ray tracing forward model - computes intersection points on display
        # DM.trace returns a tuple; [0] is the intersection list.
        # We stack them and take the first 2 dimensions (x, y).
        ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
        return ps
    elif mode == 'render':
        # Image rendering forward model - renders images from current parameters
        I = valid_cap * torch.stack(DM.render(with_element=True, angles=angle))
        I[torch.isnan(I)] = 0.0
        return I
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trace' or 'render'.")

def run_inversion(DM, ps_cap, valid_cap, angle, device, maxit=5):
    """
    Run the optimization/inversion to recover lens parameters.
    """
    # 1. Initialize lens parameters (perturbation)
    print("Initialize lens parameters (perturbation).")
    DM.scene.lensgroup.surfaces[0].c = torch.Tensor([0.00]).to(device)  # 1st surface curvature
    DM.scene.lensgroup.surfaces[1].c = torch.Tensor([0.00]).to(device)  # 2nd surface curvature
    DM.scene.lensgroup.surfaces[1].d = torch.Tensor([3.00]).to(device)  # lens thickness
    DM.scene.lensgroup.theta_x = torch.Tensor([0.00]).to(device)        # lens X-tilt angle
    DM.scene.lensgroup.theta_y = torch.Tensor([0.00]).to(device)        # lens Y-tilt angle
    DM.scene.lensgroup.update() # Propagate changes through the system

    # 2. Set optimization parameters
    diff_names = [
        'lensgroup.surfaces[0].c',
        'lensgroup.surfaces[1].c',
        'lensgroup.surfaces[1].d',
        'lensgroup.origin',
        'lensgroup.theta_x',
        'lensgroup.theta_y'
    ]

    # 3. Define Closures for the Solver
    def forward():
        # Returns current prediction based on current DM state
        return forward_operator(DM, angle, valid_cap, mode='trace')

    def loss(ps):
        # Scalar loss for backpropagation
        return torch.sum((ps[valid_cap, ...] - ps_cap[valid_cap, ...]) ** 2, axis=-1).mean()

    def func_yref_y(ps):
        # Residual vector calculation (Target - Prediction)
        b = valid_cap[..., None] * (ps_cap - ps)
        b[torch.isnan(b)] = 0.0 
        return b

    # 4. Optimize
    print(f"Running optimization with maxit={maxit}...")
    ls = DM.solve(diff_names, forward, loss, func_yref_y, option='Adam', R='I', maxit=maxit)

    return ls, DM