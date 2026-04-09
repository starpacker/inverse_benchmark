import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

import torch

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(model, x, device):
    """
    Apply the FNO forward operator to input coefficient field(s).
    
    Args:
        model: Trained FNO2d model
        x: Input tensor of shape [batch, 1, H, W] or numpy array [batch, H, W]
        device: torch device
    
    Returns:
        Predicted solution field as numpy array [batch, 1, H, W]
    """
    model.eval()
    
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        elif x.ndim == 3:
            x = x[:, np.newaxis, :, :]
        x = torch.FloatTensor(x)
    
    x = x.to(device)
    
    with torch.no_grad():
        y_pred = model(x)
    
    return y_pred.cpu().numpy()
