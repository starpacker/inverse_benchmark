import matplotlib

matplotlib.use('Agg')

import os

import sys

import torch

import torch.nn as nn

import torch.nn.functional as F

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
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
        
        self.fc0 = nn.Linear(3, self.width)
        
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        
        grid_x = torch.linspace(0, 1, size_x, device=x.device).reshape(1, 1, -1, 1).repeat(batch_size, 1, 1, size_y)
        grid_y = torch.linspace(0, 1, size_y, device=x.device).reshape(1, 1, 1, -1).repeat(batch_size, 1, size_x, 1)
        x = torch.cat([x, grid_x, grid_y], dim=1)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

def run_inversion(train_loader, test_loader, modes, width, n_layers, epochs, lr, device):
    """
    Train the FNO model to learn the mapping from coefficient field to solution field.
    This is the "inversion" in the sense that we learn the inverse mapping from
    PDE parameters to solutions.
    
    Returns:
        Trained model and training history
    """
    print(f"\n[MODEL] Building FNO2d...")
    model = FNO2d(modes1=modes, modes2=modes, width=width, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {n_params:,}")
    
    print(f"\n[TRAIN] Training FNO for {epochs} epochs...")
    
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
            
            loss = torch.mean(torch.norm(pred - u_batch, dim=(-2, -1)) / 
                            (torch.norm(u_batch, dim=(-2, -1)) + 1e-8))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
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
    
    result = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    
    return result
