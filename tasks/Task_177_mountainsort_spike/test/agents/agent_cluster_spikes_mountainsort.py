import numpy as np

import matplotlib

matplotlib.use("Agg")

def cluster_spikes_mountainsort(recording, sampling_rate, num_channels):
    """Try mountainsort5 via spikeinterface."""
    try:
        import spikeinterface.core as si
        import mountainsort5 as ms5

        # Create a NumpyRecording
        rec = si.NumpyRecording(
            traces_list=[recording],
            sampling_frequency=sampling_rate,
        )
        rec.set_dummy_probe_from_locations(
            np.array([[0, i * 50] for i in range(num_channels)]).astype(float)
        )

        sorting = ms5.sorting_scheme2(
            recording=rec,
            sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_threshold=5.0,
                phase1_detect_channel_radius=200,
                phase1_detect_time_radius_msec=0.5,
            ),
        )
        unit_ids = sorting.get_unit_ids()
        ms_spike_times = []
        ms_spike_labels = []
        for uid in unit_ids:
            st = sorting.get_unit_spike_train(uid)
            ms_spike_times.append(st)
            ms_spike_labels.append(np.full(len(st), uid))
        if len(ms_spike_times) > 0:
            ms_spike_times = np.concatenate(ms_spike_times)
            ms_spike_labels = np.concatenate(ms_spike_labels)
            order = np.argsort(ms_spike_times)
            print(f"[mountainsort5] Found {len(unit_ids)} units, {len(ms_spike_times)} spikes")
            return ms_spike_times[order], ms_spike_labels[order], "mountainsort5"
        else:
            raise RuntimeError("mountainsort5 found 0 units")
    except Exception as e:
        print(f"[mountainsort5] failed: {e}. Falling back to GMM.")
        return None, None, None
