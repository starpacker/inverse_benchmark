import matplotlib

matplotlib.use('Agg')

import numpy as np

import torch

def create_test_image(size: int, seed: int) -> np.ndarray:
    """Create a synthetic test image with geometric shapes and textures."""
    img = np.zeros((size, size), dtype=np.float64)

    # Background gradient
    yy, xx = np.mgrid[0:size, 0:size]
    img += 0.15 * (xx / size) + 0.1 * (yy / size)

    # Circles
    for cx, cy, r, v in [(35, 35, 18, 0.9), (90, 40, 14, 0.7),
                          (60, 90, 20, 0.8), (100, 100, 10, 0.6)]:
        mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        img[mask] = v

    # Rectangle
    img[20:50, 70:110] = 0.5

    # Sinusoidal texture
    img += 0.08 * np.sin(2 * np.pi * xx / 16) * np.cos(2 * np.pi * yy / 20)

    # Small bright dots (stars)
    rng = np.random.RandomState(seed)
    for _ in range(15):
        px, py = rng.randint(5, size - 5, size=2)
        img[py - 1:py + 2, px - 1:px + 2] = 0.95

    img = np.clip(img, 0.0, 1.0)
    return img

def make_blur_kernel(ksize: int, sigma: float) -> torch.Tensor:
    """Create a 2-D Gaussian blur kernel."""
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = g.outer(g)
    kernel /= kernel.sum()
    return kernel

def make_schedule(T: int, beta_start: float, beta_end: float):
    """Linear β schedule and derived quantities."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        'betas': betas.float(),
        'alphas': alphas.float(),
        'alpha_bar': alpha_bar.float(),
        'sqrt_alpha_bar': alpha_bar.sqrt().float(),
        'sqrt_one_minus_alpha_bar': (1.0 - alpha_bar).sqrt().float(),
    }

def load_and_preprocess_data(img_size: int, blur_kernel_size: int, blur_sigma: float,
                              noise_std: float, num_timesteps: int, beta_start: float,
                              beta_end: float, seed: int, device: torch.device):
    """
    Load and preprocess data for DPS deblurring.
    
    Returns:
        dict containing:
            - gt_np: ground truth image as numpy array
            - gt_tensor: ground truth image as torch tensor
            - blur_kernel: Gaussian blur kernel
            - schedule: diffusion schedule dictionary
            - config: configuration parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create ground truth image
    print("\n[1/6] Creating ground-truth image ...")
    gt_np = create_test_image(img_size, seed)
    gt_tensor = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device)
    print(f"  GT shape: {gt_np.shape}, range: [{gt_np.min():.3f}, {gt_np.max():.3f}]")
    
    # Create blur kernel
    blur_kernel = make_blur_kernel(blur_kernel_size, blur_sigma).to(device)
    
    # Create diffusion schedule
    print("\n[3/6] Setting up diffusion schedule ...")
    schedule = make_schedule(num_timesteps, beta_start, beta_end)
    print(f"  T={num_timesteps}, β ∈ [{beta_start}, {beta_end}]")
    
    config = {
        'img_size': img_size,
        'blur_kernel_size': blur_kernel_size,
        'blur_sigma': blur_sigma,
        'noise_std': noise_std,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'seed': seed,
        'device': device,
    }
    
    return {
        'gt_np': gt_np,
        'gt_tensor': gt_tensor,
        'blur_kernel': blur_kernel,
        'schedule': schedule,
        'config': config,
    }
