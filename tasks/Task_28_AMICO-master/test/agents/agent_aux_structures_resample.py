import warnings

import numpy as np

from dipy.core.geometry import cart2sphere

from dipy.reconst.shm import real_sh_descoteaux

warnings.filterwarnings("ignore")

def aux_structures_resample(scheme, lmax=12):
    nSH = int((lmax + 1) * (lmax + 2) / 2)
    idx_OUT = np.zeros(scheme.dwi_count, dtype=int)
    Ylm_OUT = np.zeros((scheme.dwi_count, nSH * len(scheme.shells)), dtype=np.float32)
    
    idx = 0
    for s in range(len(scheme.shells)):
        nS = len(scheme.shells[s]['idx'])
        idx_OUT[idx:idx+nS] = scheme.shells[s]['idx']
        
        grad = scheme.shells[s]['grad']
        _, theta, phi = cart2sphere(grad[:, 0], grad[:, 1], grad[:, 2])
        Y, _, _ = real_sh_descoteaux(lmax, theta, phi)
        
        Ylm_OUT[idx:idx+nS, nSH*s:nSH*(s+1)] = Y
        idx += nS
    return idx_OUT, Ylm_OUT
