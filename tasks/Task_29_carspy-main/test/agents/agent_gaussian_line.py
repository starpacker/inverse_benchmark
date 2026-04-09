import numpy as np

def gaussian_line(w, w0, sigma):
    if sigma == 0: return np.zeros_like(w)
    return 2/sigma*(np.log(2)/np.pi)**0.5*np.exp(-4*np.log(2)*((w-w0)/sigma)**2)
