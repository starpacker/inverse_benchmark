import numpy as np

import matplotlib

matplotlib.use('Agg')

def pad2d(data, nx, ny):
    datsize = data.shape
    padsize = np.array(datsize)
    padsize[0] = nx
    padsize[1] = ny
    ndata = np.zeros(tuple(padsize), dtype=data.dtype)
    datrx = int(datsize[0]/2)
    datry = int(datsize[1]/2)
    cx = int(nx/2)
    cy = int(ny/2)
    cxr = np.arange(round(cx-datrx), round(cx-datrx+datsize[0]))
    cyr = np.arange(round(cy-datry), round(cy-datry+datsize[1]))
    ndata[np.ix_(cxr.astype(int), cyr.astype(int))] = data
    return ndata
