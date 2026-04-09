import matplotlib

matplotlib.use('Agg')

import numpy as np

def oasis_trace(F, tau, fs):
    """
    OASIS: Online Active Set method for Spike Inference on a single trace.
    
    Implements the pool-adjacent-violators algorithm for non-negative
    deconvolution of calcium signals with AR(1) model: c_t = g * c_{t-1} + s_t.
    """
    g = np.exp(-1.0 / (tau * fs))
    NT = len(F)
    
    v = np.zeros(NT, dtype=np.float64)
    w = np.zeros(NT, dtype=np.float64)
    t = np.zeros(NT, dtype=np.int64)
    l = np.zeros(NT, dtype=np.int64)
    
    it = 0
    ip = 0
    
    while it < NT:
        v[ip] = F[it]
        w[ip] = 1.0
        t[ip] = it
        l[ip] = 1
        
        while ip > 0:
            if v[ip - 1] * (g ** l[ip - 1]) > v[ip]:
                f1 = g ** l[ip - 1]
                f2 = g ** (2 * l[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                l[ip - 1] = l[ip - 1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1
    
    s = np.zeros(NT, dtype=np.float64)
    n_pools = ip
    for i in range(1, n_pools):
        spike_val = v[i] - v[i - 1] * (g ** l[i - 1])
        if spike_val > 0:
            s[t[i]] = spike_val
    if n_pools > 0 and v[0] > 0:
        s[t[0]] = v[0]
    
    c = np.zeros(NT, dtype=np.float64)
    c[0] = s[0]
    for i in range(1, NT):
        c[i] = g * c[i - 1] + s[i]
    
    return c, s
