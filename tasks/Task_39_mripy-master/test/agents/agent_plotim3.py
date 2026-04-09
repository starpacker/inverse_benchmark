import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import matplotlib.cm as cm

def plotim3(im, save_path=None):
    im = np.flip(im, 0)
    plt.figure()
    plt.imshow(im, cmap=cm.gray, origin='lower', interpolation='none')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.close()
