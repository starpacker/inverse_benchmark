import warnings

warnings.filterwarnings("ignore")

def aux_structures_generate(scheme, lmax=12):
    nSH = int((lmax + 1) * (lmax + 2) / 2)
    idx_IN = []
    idx_OUT = []
    for s in range(len(scheme.shells)):
        idx_IN.append(range(500 * s, 500 * (s + 1)))
        idx_OUT.append(range(nSH * s, nSH * (s + 1)))
    return idx_IN, idx_OUT
