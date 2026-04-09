"""
neuralop_fno - Fourier Neural Operator for PDE Inverse Problems
===============================================================
Task: Learn PDE solution operator for fast inverse problem solving (Darcy Flow)
Repo: https://github.com/neuraloperator/neuraloperator

The Fourier Neural Operator (FNO) learns the mapping from PDE coefficients
(permeability field) to PDE solutions (pressure field) for the 2D Darcy flow
equation: -∇·(a(x)∇u(x)) = f(x).

Usage:
    /data/yjh/neuralop_fno_env/bin/python neuralop_fno_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import time

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# FNO hyperparameters
RESOLUTION = 64          # Grid resolution
N_TRAIN = 400            # Training samples
N_TEST = 50              # Test samples
MODES = 12               # Fourier modes to keep
WIDTH = 32               # Channel width
N_LAYERS = 4             # Number of Fourier layers
EPOCHS = 100             # Training epochs
BATCH_SIZE = 20          # Batch size
LR = 1e-3                # Learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════
# 2. Data Generation - Darcy Flow PDE
# ═══════════════════════════════════════════════════════════
def generate_gaussian_random_field(n, resolution, alpha=2.0, tau=3.0, seed=None):
    """Generate Gaussian Random Field for permeability coefficients."""
    if seed is not None:
        np.random.seed(seed)
    
    fields = []
    for _ in range(n):
        # Generate in frequency domain
        k1 = np.fft.fftfreq(resolution, d=1.0/resolution)
        k2 = np.fft.fftfreq(resolution, d=1.0/resolution)
        K1, K2 = np.meshgrid(k1, k2)
        
        # Power spectrum: (tau^2 + |k|^2)^(-alpha)
        power = (tau**2 + K1**2 + K2**2)**(-alpha/2.0)
        power[0, 0] = 0  # Zero mean
        
        # Random Fourier coefficients
        coeff_real = np.random.randn(resolution, resolution)
        coeff_imag = np.random.randn(resolution, resolution)
        coeff = (coeff_real + 1j * coeff_imag) * power
        
        # Transform to spatial domain
        field = np.real(np.fft.ifft2(coeff * resolution))
        
        # Make positive (permeability must be positive)
        # Use softplus-like transform
        field = np.exp(field)
        # Normalize to [3, 12] range (typical Darcy flow permeability)
        field = 3 + 9 * (field - field.min()) / (field.max() - field.min() + 1e-8)
        
        fields.append(field)
    
    return np.array(fields, dtype=np.float32)


def solve_darcy_flow(a, f_source=1.0):
    """
    Solve 2D Darcy flow: -∇·(a(x)∇u(x)) = f on [0,1]^2
    with u = 0 on boundary (Dirichlet BC).
    Uses finite differences with 5-point stencil.
    """
    n = a.shape[0]
    h = 1.0 / (n + 1)  # Grid spacing
    
    # Interior points only (boundary is 0)
    N = n * n
    
    # Build sparse system using vectorized operations
    # -∇·(a∇u) ≈ finite difference stencil
    # We'll use a simpler approach: iterative Gauss-Seidel solver
    
    u = np.zeros((n, n), dtype=np.float64)
    f = np.ones((n, n), dtype=np.float64) * f_source
    
    # Gauss-Seidel iterations
    for iteration in range(500):
        u_old = u.copy()
        
        for i in range(1, n-1):
            for j in range(1, n-1):
                # Harmonic mean of coefficients for interface
                a_right = 2 * a[i, j] * a[i, j+1] / (a[i, j] + a[i, j+1] + 1e-10)
                a_left = 2 * a[i, j] * a[i, j-1] / (a[i, j] + a[i, j-1] + 1e-10)
                a_up = 2 * a[i, j] * a[i-1, j] / (a[i, j] + a[i-1, j] + 1e-10)
                a_down = 2 * a[i, j] * a[i+1, j] / (a[i, j] + a[i+1, j] + 1e-10)
                
                denom = a_right + a_left + a_up + a_down
                numer = (a_right * u[i, j+1] + a_left * u[i, j-1] + 
                        a_up * u[i-1, j] + a_down * u[i+1, j] + h**2 * f[i, j])
                
                u[i, j] = numer / (denom + 1e-10)
        
        # Check convergence
        if np.max(np.abs(u - u_old)) < 1e-6:
            break
    
    return u.astype(np.float32)


def solve_darcy_flow_fast(a, f_source=1.0):
    """
    Fast Darcy flow solver using scipy sparse solver.
    -∇·(a(x)∇u(x)) = f on [0,1]^2, u=0 on boundary.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    n = a.shape[0]
    h = 1.0 / (n + 1)
    N = n * n
    
    # Build sparse matrix
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            rhs[idx] = f_source * h**2
            
            center_val = 0.0
            
            # Right neighbor
            if j < n - 1:
                a_e = 0.5 * (a[i, j] + a[i, j+1])
                rows.append(idx); cols.append(idx + 1); vals.append(a_e)
                center_val += a_e
            else:
                center_val += a[i, j]  # boundary
                
            # Left neighbor
            if j > 0:
                a_w = 0.5 * (a[i, j] + a[i, j-1])
                rows.append(idx); cols.append(idx - 1); vals.append(a_w)
                center_val += a_w
            else:
                center_val += a[i, j]
                
            # Down neighbor
            if i < n - 1:
                a_s = 0.5 * (a[i, j] + a[i+1, j])
                rows.append(idx); cols.append(idx + n); vals.append(a_s)
                center_val += a_s
            else:
                center_val += a[i, j]
                
            # Up neighbor
            if i > 0:
                a_n = 0.5 * (a[i, j] + a[i-1, j])
                rows.append(idx); cols.append(idx - n); vals.append(a_n)
                center_val += a_n
            else:
                center_val += a[i, j]
            
            rows.append(idx); cols.append(idx); vals.append(-center_val)
    
    A_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u_vec = spsolve(A_mat, -rhs)
    u = u_vec.reshape(n, n).astype(np.float32)
    
    return u


def generate_darcy_dataset(n_samples, resolution, seed=42):
    """Generate Darcy flow dataset: (permeability_field, solution_field) pairs."""
    print(f"[DATA] Generating {n_samples} Darcy flow samples at {resolution}×{resolution}...")
    
    # Generate random permeability fields
    a_fields = generate_gaussian_random_field(n_samples, resolution, alpha=2.5, tau=5.0, seed=seed)
    
    # Solve PDE for each field
    u_fields = np.zeros_like(a_fields)
    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Solving PDE {i+1}/{n_samples}...")
        u_fields[i] = solve_darcy_flow_fast(a_fields[i])
    
    # Normalize solutions
    u_max = np.max(np.abs(u_fields))
    if u_max > 0:
        u_fields = u_fields / u_max
    
    return a_fields, u_fields


# ═══════════════════════════════════════════════════════════
# 3. Fourier Neural Operator (FNO) Model
# ═══════════════════════════════════════════════════════════
class SpectralConv2d(nn.Module):
    """2D Fourier layer: performs spectral convolution."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    
    def compl_mul2d(self, input, weights):
        """Complex multiplication: (batch, in, x, y) × (in, out, x, y) → (batch, out, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D problems.
    Input: coefficient field a(x) [batch, 1, H, W]
    Output: solution field u(x) [batch, 1, H, W]
    """
    def __init__(self, modes1, modes2, width, n_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Lifting: input channels (1 + 2 for grid) → width
        self.fc0 = nn.Linear(3, self.width)
        
        # Fourier layers
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))
        
        # Projection: width → 128 → 1
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # x: [batch, 1, H, W]
        batch_size = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        
        # Add grid coordinates
        grid_x = torch.linspace(0, 1, size_x, device=x.device).reshape(1, 1, -1, 1).repeat(batch_size, 1, 1, size_y)
        grid_y = torch.linspace(0, 1, size_y, device=x.device).reshape(1, 1, 1, -1).repeat(batch_size, 1, size_x, 1)
        x = torch.cat([x, grid_x, grid_y], dim=1)  # [batch, 3, H, W]
        
        # Reshape for linear layer: [batch, H, W, 3]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)  # [batch, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, H, W]
        
        # Fourier layers
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # Projection
        x = x.permute(0, 2, 3, 1)  # [batch, H, W, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # [batch, H, W, 1]
        x = x.permute(0, 3, 1, 2)  # [batch, 1, H, W]
        
        return x


# ═══════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════
def train_fno(model, train_loader, test_loader, epochs, lr, device):
    """Train FNO model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for a_batch, u_batch in train_loader:
            a_batch = a_batch.to(device)
            u_batch = u_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(a_batch)
            
            # Relative L2 loss
            loss = torch.mean(torch.norm(pred - u_batch, dim=(-2, -1)) / 
                            (torch.norm(u_batch, dim=(-2, -1)) + 1e-8))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            test_loss = 0.0
            n_test = 0
            with torch.no_grad():
                for a_batch, u_batch in test_loader:
                    a_batch = a_batch.to(device)
                    u_batch = u_batch.to(device)
                    pred = model(a_batch)
                    rel_err = torch.mean(torch.norm(pred - u_batch, dim=(-2, -1)) / 
                                       (torch.norm(u_batch, dim=(-2, -1)) + 1e-8))
                    test_loss += rel_err.item()
                    n_test += 1
            avg_test_loss = test_loss / n_test
            test_losses.append(avg_test_loss)
            print(f"  Epoch {epoch+1}/{epochs}: Train L2={avg_train_loss:.4f}, Test L2={avg_test_loss:.4f}")
    
    return train_losses, test_losses


# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_psnr(ref, test, data_range=None):
    """Compute PSNR (dB)."""
    if data_range is None:
        data_range = ref.max() - ref.min()
    mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)

def compute_ssim(ref, test):
    """Compute SSIM."""
    from skimage.metrics import structural_similarity as ssim
    data_range = ref.max() - ref.min()
    if data_range == 0:
        data_range = 1.0
    return ssim(ref, test, data_range=data_range)

def compute_rmse(ref, test):
    """Compute RMSE."""
    return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

def compute_relative_l2(ref, test):
    """Compute relative L2 error."""
    return np.linalg.norm(ref - test) / (np.linalg.norm(ref) + 1e-10)


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(coeff_field, gt_solution, fno_prediction, metrics, save_path):
    """Generate 4-panel visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Permeability field (input)
    im0 = axes[0].imshow(coeff_field, cmap='viridis', origin='lower')
    axes[0].set_title('Input: Permeability a(x)', fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Panel 2: Ground truth solution
    im1 = axes[1].imshow(gt_solution, cmap='RdBu_r', origin='lower')
    axes[1].set_title('GT: PDE Solution u(x)', fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Panel 3: FNO prediction
    im2 = axes[2].imshow(fno_prediction, cmap='RdBu_r', origin='lower',
                         vmin=gt_solution.min(), vmax=gt_solution.max())
    axes[2].set_title('FNO Prediction', fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Panel 4: Error map
    error = np.abs(gt_solution - fno_prediction)
    im3 = axes[3].imshow(error, cmap='hot', origin='lower')
    axes[3].set_title('|Error|', fontsize=12)
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    fig.suptitle(
        f"FNO Darcy Flow | PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"Rel. L2={metrics['relative_l2']:.4f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  neuralop_fno — Fourier Neural Operator for Darcy Flow")
    print("=" * 60)
    print(f"[CONFIG] Device: {DEVICE}")
    print(f"[CONFIG] Resolution: {RESOLUTION}, Modes: {MODES}, Width: {WIDTH}")
    print(f"[CONFIG] Train: {N_TRAIN}, Test: {N_TEST}, Epochs: {EPOCHS}")
    
    t0 = time.time()
    
    # (a) Generate data
    a_fields, u_fields = generate_darcy_dataset(N_TRAIN + N_TEST, RESOLUTION, seed=42)
    
    a_train, a_test = a_fields[:N_TRAIN], a_fields[N_TRAIN:]
    u_train, u_test = u_fields[:N_TRAIN], u_fields[N_TRAIN:]
    
    print(f"[DATA] Coefficient fields shape: {a_train.shape}")
    print(f"[DATA] Solution fields shape: {u_train.shape}")
    
    # Convert to tensors
    a_train_t = torch.FloatTensor(a_train).unsqueeze(1)  # [N, 1, H, W]
    u_train_t = torch.FloatTensor(u_train).unsqueeze(1)
    a_test_t = torch.FloatTensor(a_test).unsqueeze(1)
    u_test_t = torch.FloatTensor(u_test).unsqueeze(1)
    
    train_dataset = TensorDataset(a_train_t, u_train_t)
    test_dataset = TensorDataset(a_test_t, u_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # (b) Build and train FNO
    print(f"\n[MODEL] Building FNO2d...")
    model = FNO2d(modes1=MODES, modes2=MODES, width=WIDTH, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {n_params:,}")
    
    print(f"\n[TRAIN] Training FNO for {EPOCHS} epochs...")
    train_losses, test_losses = train_fno(model, train_loader, test_loader, EPOCHS, LR, DEVICE)
    
    # (c) Evaluate on test set
    print(f"\n[EVAL] Evaluating on test set...")
    model.eval()
    
    all_psnr, all_ssim, all_rmse, all_rel_l2 = [], [], [], []
    all_preds = []
    
    with torch.no_grad():
        for a_batch, u_batch in test_loader:
            a_batch = a_batch.to(DEVICE)
            pred = model(a_batch)
            all_preds.append(pred.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)  # [N_test, 1, H, W]
    
    for i in range(N_TEST):
        gt_i = u_test[i]
        pred_i = predictions[i, 0]
        
        all_psnr.append(compute_psnr(gt_i, pred_i))
        all_ssim.append(compute_ssim(gt_i, pred_i))
        all_rmse.append(compute_rmse(gt_i, pred_i))
        all_rel_l2.append(compute_relative_l2(gt_i, pred_i))
    
    metrics = {
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "rmse": float(np.mean(all_rmse)),
        "relative_l2": float(np.mean(all_rel_l2)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_std": float(np.std(all_ssim)),
        "n_test": N_TEST,
        "n_train": N_TRAIN,
        "epochs": EPOCHS,
        "device": str(DEVICE),
    }
    
    print(f"[EVAL] Mean PSNR = {metrics['psnr']:.4f} dB (±{metrics['psnr_std']:.2f})")
    print(f"[EVAL] Mean SSIM = {metrics['ssim']:.6f} (±{metrics['ssim_std']:.4f})")
    print(f"[EVAL] Mean RMSE = {metrics['rmse']:.6f}")
    print(f"[EVAL] Mean Rel. L2 = {metrics['relative_l2']:.6f}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")
    
    # (e) Visualize best test sample
    best_idx = int(np.argmax(all_psnr))
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    sample_metrics = {
        "psnr": all_psnr[best_idx],
        "ssim": all_ssim[best_idx],
        "rmse": all_rmse[best_idx],
        "relative_l2": all_rel_l2[best_idx],
    }
    visualize_results(a_test[best_idx], u_test[best_idx], predictions[best_idx, 0],
                     sample_metrics, vis_path)
    
    # (f) Save arrays (best test sample)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), predictions[best_idx, 0])
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), u_test[best_idx])
    np.save(os.path.join(RESULTS_DIR, "input_coefficient.npy"), a_test[best_idx])
    
    elapsed = time.time() - t0
    print(f"\n[TIME] Total elapsed: {elapsed:.1f}s")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
