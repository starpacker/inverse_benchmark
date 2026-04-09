import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

import torch

from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_gaussian_random_field(n, resolution, alpha=2.0, tau=3.0, seed=None):
    """Generate Gaussian Random Field for permeability coefficients."""
    if seed is not None:
        np.random.seed(seed)
    
    fields = []
    for _ in range(n):
        k1 = np.fft.fftfreq(resolution, d=1.0/resolution)
        k2 = np.fft.fftfreq(resolution, d=1.0/resolution)
        K1, K2 = np.meshgrid(k1, k2)
        
        power = (tau**2 + K1**2 + K2**2)**(-alpha/2.0)
        power[0, 0] = 0
        
        coeff_real = np.random.randn(resolution, resolution)
        coeff_imag = np.random.randn(resolution, resolution)
        coeff = (coeff_real + 1j * coeff_imag) * power
        
        field = np.real(np.fft.ifft2(coeff * resolution))
        field = np.exp(field)
        field = 3 + 9 * (field - field.min()) / (field.max() - field.min() + 1e-8)
        
        fields.append(field)
    
    return np.array(fields, dtype=np.float32)

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
    
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            rhs[idx] = f_source * h**2
            
            center_val = 0.0
            
            if j < n - 1:
                a_e = 0.5 * (a[i, j] + a[i, j+1])
                rows.append(idx); cols.append(idx + 1); vals.append(a_e)
                center_val += a_e
            else:
                center_val += a[i, j]
                
            if j > 0:
                a_w = 0.5 * (a[i, j] + a[i, j-1])
                rows.append(idx); cols.append(idx - 1); vals.append(a_w)
                center_val += a_w
            else:
                center_val += a[i, j]
                
            if i < n - 1:
                a_s = 0.5 * (a[i, j] + a[i+1, j])
                rows.append(idx); cols.append(idx + n); vals.append(a_s)
                center_val += a_s
            else:
                center_val += a[i, j]
                
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

def load_and_preprocess_data(n_train, n_test, resolution, batch_size, seed=42):
    """
    Generate Darcy flow dataset: (permeability_field, solution_field) pairs.
    Returns training and test DataLoaders, along with raw numpy arrays for evaluation.
    """
    print(f"[DATA] Generating {n_train + n_test} Darcy flow samples at {resolution}×{resolution}...")
    
    n_samples = n_train + n_test
    a_fields = generate_gaussian_random_field(n_samples, resolution, alpha=2.5, tau=5.0, seed=seed)
    
    u_fields = np.zeros_like(a_fields)
    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Solving PDE {i+1}/{n_samples}...")
        u_fields[i] = solve_darcy_flow_fast(a_fields[i])
    
    u_max = np.max(np.abs(u_fields))
    if u_max > 0:
        u_fields = u_fields / u_max
    
    a_train, a_test = a_fields[:n_train], a_fields[n_train:]
    u_train, u_test = u_fields[:n_train], u_fields[n_train:]
    
    print(f"[DATA] Coefficient fields shape: {a_train.shape}")
    print(f"[DATA] Solution fields shape: {u_train.shape}")
    
    a_train_t = torch.FloatTensor(a_train).unsqueeze(1)
    u_train_t = torch.FloatTensor(u_train).unsqueeze(1)
    a_test_t = torch.FloatTensor(a_test).unsqueeze(1)
    u_test_t = torch.FloatTensor(u_test).unsqueeze(1)
    
    train_dataset = TensorDataset(a_train_t, u_train_t)
    test_dataset = TensorDataset(a_test_t, u_test_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    data_dict = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'a_train': a_train,
        'u_train': u_train,
        'a_test': a_test,
        'u_test': u_test,
        'a_test_tensor': a_test_t,
        'u_test_tensor': u_test_t,
    }
    
    return data_dict
