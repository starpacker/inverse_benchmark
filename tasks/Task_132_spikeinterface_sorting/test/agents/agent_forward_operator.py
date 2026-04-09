import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(
    sorting_gt,
    recording,
    bin_size_ms: float = 1.0,
) -> np.ndarray:
    """
    Forward model: Convert spike trains to a binary raster representation.
    
    The forward model describes how each neuron's spike train contributes
    to the observed signal. Here we represent the spike trains as a binary
    matrix (units x time_bins) where 1 indicates a spike occurred.
    
    Args:
        sorting_gt: Ground truth sorting object with spike trains
        recording: Recording object (for timing info)
        bin_size_ms: Size of time bins in milliseconds
        
    Returns:
        gt_raster: Binary matrix of shape (num_units, num_bins)
    """
    sampling_frequency = recording.get_sampling_frequency()
    num_samples = recording.get_num_samples()
    
    bin_size_samples = int(bin_size_ms * sampling_frequency / 1000.0)
    n_bins = int(np.ceil(num_samples / bin_size_samples))
    
    gt_raster = np.zeros((sorting_gt.get_num_units(), n_bins), dtype=np.float32)
    
    for i, uid in enumerate(sorting_gt.unit_ids):
        spike_train = sorting_gt.get_unit_spike_train(uid)
        bins = spike_train // bin_size_samples
        bins = bins[bins < n_bins]
        gt_raster[i, bins] = 1.0
    
    return gt_raster
