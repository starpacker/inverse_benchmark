import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.spatial.distance import cdist

def _match_particles(gt, det, md):
    """Match detected particles to ground truth within distance md."""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    D = cdist(gt, det)
    mg = []
    md_ = []
    ug = set()
    ud = set()
    for idx in np.argsort(D, axis=None):
        gi, di = np.unravel_index(idx, D.shape)
        if gi in ug or di in ud:
            continue
        if D[gi, di] > md:
            break
        mg.append(gt[gi])
        md_.append(det[di])
        ug.add(gi)
        ud.add(di)
    return np.array(mg), np.array(md_)
