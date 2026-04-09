import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import time

import shutil

def run_inversion(
    recording_cmr,
    working_dir: str,
) -> tuple:
    """
    Run spike sorting inversion to recover spike trains from the recording.
    
    The inverse problem: Given the multi-channel recording V(t), recover
    the spike trains {t_ij} for each neuron. This is a blind source
    separation problem.
    
    Args:
        recording_cmr: Preprocessed recording
        working_dir: Directory for output files
        
    Returns:
        sorting: Detected sorting object
        sorter_used: Name of the sorter that was used
    """
    import spikeinterface.full as si
    
    print("\n[3/7] Running spike sorting...")
    t0 = time.time()
    
    sorting_output = os.path.join(working_dir, 'sorting_output')
    if os.path.exists(sorting_output):
        shutil.rmtree(sorting_output)
    
    sorting = None
    sorter_used = None
    
    # Try spykingcircus2 first, then tridesclous2
    for sorter_name in ['spykingcircus2', 'tridesclous2']:
        try:
            print(f"  Trying sorter: {sorter_name}")
            out_folder = sorting_output + f"_{sorter_name}"
            if os.path.exists(out_folder):
                shutil.rmtree(out_folder)
            sorting = si.run_sorter(
                sorter_name,
                recording_cmr,
                folder=out_folder,
                verbose=True,
                remove_existing_folder=True,
            )
            sorter_used = sorter_name
            print(f"  SUCCESS with {sorter_name}")
            break
        except Exception as e:
            print(f"  {sorter_name} failed: {e}")
            continue
    
    if sorting is None:
        print("  Built-in sorters failed. Using manual lightweight approach...")
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.clustering import find_clusters_from_peaks
        
        # Detect peaks
        print("  Detecting peaks...")
        peaks = detect_peaks(
            recording_cmr,
            method='locally_exclusive',
            peak_sign='neg',
            detect_threshold=5,
            noise_levels=None,
            random_chunk_kwargs={},
            outputs='numpy_compact',
            pipeline_nodes=None,
            gather_mode='memory',
            job_kwargs=dict(n_jobs=1, chunk_duration='1s'),
        )
        print(f"  Found {len(peaks)} peaks")
        
        # Cluster peaks
        print("  Clustering peaks...")
        labels, peak_labels = find_clusters_from_peaks(
            recording_cmr,
            peaks,
            method='hdbscan',
            method_kwargs={},
            job_kwargs=dict(n_jobs=1),
        )
        
        # Create a NumpySorting from labels
        from spikeinterface.core import NumpySorting
        unique_labels = np.unique(peak_labels[peak_labels >= 0])
        spike_trains = {}
        for label in unique_labels:
            mask = peak_labels == label
            spike_trains[int(label)] = peaks['sample_index'][mask]
        
        sorting = NumpySorting.from_dict(
            spike_trains,
            sampling_frequency=recording_cmr.get_sampling_frequency(),
        )
        sorter_used = "manual_peak_clustering"
    
    print(f"  Sorter used: {sorter_used}")
    print(f"  Detected units: {sorting.get_num_units()}")
    for uid in sorting.unit_ids:
        n_spikes = len(sorting.get_unit_spike_train(uid))
        print(f"    Unit {uid}: {n_spikes} spikes")
    print(f"  Sorting took {time.time() - t0:.1f}s")
    
    return sorting, sorter_used
