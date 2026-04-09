import numpy as np

def asym_Gaussian(w, w0, sigma, k, a_sigma, a_k, offset, power_factor=1.):
    response_low = np.exp(-abs((w[w <= w0]-w0)/(sigma-a_sigma))**(k-a_k))
    response_high = np.exp(-abs((w[w > w0]-w0)/(sigma+a_sigma))**(k+a_k))
    response = (np.append(response_low, response_high) + offset)**power_factor
    max_val = response.max()
    if max_val == 0: return response
    return np.nan_to_num(response/max_val)
