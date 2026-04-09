import numpy as np

import matplotlib

matplotlib.use('Agg')

def mask2d(nx, ny, center_r=15, undersampling=0.5):
    k = int(round(nx*ny*undersampling))
    ri = np.random.choice(nx*ny, k, replace=False)
    ma = np.zeros(nx*ny)
    ma[ri] = 1
    mask = ma.reshape((nx, ny))
    if center_r > 0:
        cx = int(nx/2)
        cy = int(ny/2)
        cxr = np.arange(round(cx-center_r), round(cx+center_r+1))
        cyr = np.arange(round(cy-center_r), round(cy+center_r+1))
        mask[np.ix_(cxr.astype(int), cyr.astype(int))] = np.ones((cxr.shape[0], cyr.shape[0]))
    return mask
