import numpy as np

def _qggmrf_derivative(delta, sigma_x, p, q, T):
    """
    Computes the derivative of the Q-GGMRF potential function w.r.t delta.
    Approximation logic preserved from input code.
    """
    abs_d = np.abs(delta) + 1e-6 
    sign_d = np.sign(delta)
    u = abs_d / T
    
    # rho(x) = |x|^p / (1 + |x/T|^(p-q))
    # Using the derivation logic provided in the input context:
    num = abs_d ** p
    den = 1 + u ** (p - q)
    
    d_num = p * abs_d ** (p - 1) * sign_d
    d_den = (p - q) * (u ** (p - q - 1)) * (1.0/T) * sign_d
    
    grad = (d_num * den - num * d_den) / (den ** 2)
    return grad / (sigma_x ** p)

def _compute_prior_gradient(image, sigma_x, p, q, T):
    """Computes the gradient of the Q-GGMRF prior (Markov Random Field)."""
    total_grad = np.zeros_like(image)
    
    # Right neighbor interaction: x[r,c] - x[r,c+1]
    d = image - np.roll(image, -1, axis=1)
    d[:, -1] = 0 # Boundary
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Left neighbor interaction: x[r,c] - x[r,c-1]
    d = image - np.roll(image, 1, axis=1)
    d[:, 0] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Down neighbor interaction: x[r,c] - x[r+1,c]
    d = image - np.roll(image, -1, axis=0)
    d[-1, :] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    # Up neighbor interaction: x[r,c] - x[r-1,c]
    d = image - np.roll(image, 1, axis=0)
    d[0, :] = 0
    total_grad += _qggmrf_derivative(d, sigma_x, p, q, T)
    
    return total_grad
