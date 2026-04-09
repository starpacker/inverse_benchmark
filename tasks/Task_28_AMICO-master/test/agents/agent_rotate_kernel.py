import warnings

import numpy as np

warnings.filterwarnings("ignore")

def rotate_kernel(K, AUX, idx_IN, idx_OUT, is_isotropic, ndirs):
    Klm = []
    for s in range(len(idx_IN)):
        Klm.append(np.dot(AUX['fit'], K[idx_IN[s]]))
        
    n = len(idx_IN) * AUX['fit'].shape[0]
    
    if not is_isotropic:
        KRlm = np.zeros((ndirs, n), dtype=np.float32)
        for i in range(ndirs):
            Ylm_rot = AUX['Ylm_rot'][i]
            for s in range(len(idx_IN)):
                KRlm[i, idx_OUT[s]] = AUX['const'] * Klm[s][AUX['idx_m0']] * Ylm_rot
    else:
        KRlm = np.zeros(n, dtype=np.float32)
        for s in range(len(idx_IN)):
            KRlm[idx_OUT[s]] = Klm[s]
            
    return KRlm
