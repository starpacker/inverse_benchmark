import numpy as np

import matplotlib

matplotlib.use('Agg')

def crop2d(data, center_r=15):
    nx, ny = data.shape[0:2]
    if center_r > 0:
        cx = int(nx/2)
        cy = int(ny/2)
        cxr = np.arange(round(cx-center_r), round(cx+center_r))
        cyr = np.arange(round(cy-center_r), round(cy+center_r))
    return data[np.ix_(cxr.astype(int), cyr.astype(int))]
