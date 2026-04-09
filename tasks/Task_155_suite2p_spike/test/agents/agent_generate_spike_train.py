import matplotlib

matplotlib.use('Agg')

import numpy as np

def generate_spike_train(n_frames, spike_rate, fs, rng):
    """
    Generate a Poisson spike train.
    """
    prob_per_frame = spike_rate / fs
    spikes = (rng.random(n_frames) < prob_per_frame).astype(np.float64)
    return spikes
