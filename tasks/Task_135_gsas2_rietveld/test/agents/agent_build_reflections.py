import numpy as np

import matplotlib

matplotlib.use('Agg')

def build_reflections():
    """Build reflection table for CeO2 Fm-3m structure."""
    f_Ce, f_O = 58.0, 8.0
    refs = []
    for h in range(11):
        for k in range(h + 1):
            for ll in range(k + 1):
                if h == k == ll == 0:
                    continue
                if not (h % 2 == k % 2 == ll % 2):
                    continue
                fcc = [0, np.pi*(h+k), np.pi*(h+ll), np.pi*(k+ll)]
                fcc_sum = sum(np.exp(1j*p) for p in fcc)
                p1 = 2*np.pi*(h+k+ll)*0.25
                p2 = 2*np.pi*(h+k+ll)*0.75
                F = f_Ce*fcc_sum + f_O*fcc_sum*(np.exp(1j*p1) + np.exp(1j*p2))
                F2 = float(np.abs(F)**2)
                if F2 < 1:
                    continue
                if h == k == ll:
                    mult = 8
                elif k == 0 and ll == 0:
                    mult = 6
                elif h == k and ll == 0:
                    mult = 12
                elif h == k or k == ll:
                    mult = 24
                elif ll == 0:
                    mult = 24
                else:
                    mult = 48
                q2 = h*h + k*k + ll*ll
                refs.append((q2, F2, mult))
    return refs
