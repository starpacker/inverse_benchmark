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
