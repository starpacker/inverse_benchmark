import numpy as np

def downsample(w, w_fine, spec_fine, mode='local_mean'):
    downsampled = []
    if mode == 'interp':
        downsampled = np.interp(w, w_fine, spec_fine)
    elif mode == 'local_mean':
        hw = int((w[1] - w[0])/(w_fine[1] - w_fine[0])/2)
        if hw < 1: hw = 1
        w_fine = np.array(w_fine)
        idx = np.searchsorted(w_fine, w)
        idx[idx >= len(w_fine)] = len(w_fine) - 1
        idx[idx < 0] = 0
        
        downsampled = []
        for i in idx:
            start = max(0, i - hw)
            end = min(len(spec_fine), i + hw + 1)
            if start >= end:
                downsampled.append(spec_fine[i])
            else:
                downsampled.append(np.mean(spec_fine[start:end]))
        downsampled = np.array(downsampled)
    return downsampled
