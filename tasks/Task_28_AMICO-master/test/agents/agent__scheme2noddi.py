import warnings

import numpy as np

warnings.filterwarnings("ignore")

_GAMMA = 2.675987e8

def _scheme2noddi(scheme):
    protocol = {}
    protocol['pulseseq'] = 'PGSE'
    bval = scheme.b.copy()
    protocol['totalmeas'] = len(bval)
    protocol['b0_Indices'] = np.nonzero(bval <= 0)[0]
    protocol['numZeros'] = len(protocol['b0_Indices'])
    
    b_rounded = np.round(bval, -2)
    B = np.unique(b_rounded[b_rounded > 0])
    
    protocol['M'] = len(B)
    protocol['grad_dirs'] = scheme.raw[:, 0:3].copy()
    
    Gmax = 0.04
    maxB = np.max(B) if len(B) > 0 else 1000
    tmp = np.power(3 * maxB * 1E6 / (2 * _GAMMA**2 * Gmax**2), 1.0/3.0)
    
    protocol['delta'] = np.zeros(bval.shape)
    protocol['smalldel'] = np.zeros(bval.shape)
    protocol['gradient_strength'] = np.zeros(bval.shape)
    
    if scheme.version == 1:
        protocol['delta'] = scheme.Delta
        protocol['smalldel'] = scheme.delta
        protocol['gradient_strength'] = scheme.g
    else:
        for i, b in enumerate(B):
            mask = b_rounded == b
            protocol['delta'][mask] = tmp
            protocol['smalldel'][mask] = tmp
            protocol['gradient_strength'][mask] = np.sqrt(b / maxB) * Gmax
    return protocol
