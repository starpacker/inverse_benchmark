import warnings

import numpy as np

warnings.filterwarnings("ignore")

def resample_kernel(KRlm, nS, idx_out, Ylm_out, is_isotropic, ndirs):
    if not is_isotropic:
        KR = np.ones((ndirs, nS), dtype=np.float32)
        for i in range(ndirs):
            KR[i, idx_out] = np.dot(Ylm_out, KRlm[i, :])
    else:
        KR = np.ones(nS, dtype=np.float32)
        KR[idx_out] = np.dot(Ylm_out, KRlm)
    return KR
