import numpy as np

import matplotlib

matplotlib.use('Agg')

def hankelnd_r(a, win_shape, win_strides=None):
    if win_strides is None:
        win_strides = np.ones(win_shape.__len__()).astype(int)
    win_shape = np.array(win_shape)
    win_strides = np.array(win_strides)
    a_shape = np.array(a.shape)
    a_strides = np.array(a.strides)
    bh_shape = np.concatenate((win_shape, np.divide(a_shape - win_shape, win_strides).astype(int) + 1))
    bh_strides = np.concatenate((a_strides, np.multiply(win_strides, a_strides)))
    return np.lib.stride_tricks.as_strided(a, shape=bh_shape, strides=bh_strides)
