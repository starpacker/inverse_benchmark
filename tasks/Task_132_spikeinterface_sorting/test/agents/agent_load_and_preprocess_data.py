import matplotlib

matplotlib.use('Agg')

import time

def load_and_preprocess_data(
    durations: list,
    sampling_frequency: float,
    num_channels: int,
    num_units: int,
    firing_rates: float,
    refractory_period_ms: float,
    noise_levels: float,
    seed: int,
    freq_min: float,
    freq_max: float,
) -> tuple:
    """
    Generate simulated ground-truth extracellular recording and preprocess it.
    
    Returns:
        recording_cmr: Preprocessed recording (bandpass + common reference)
        recording_raw: Raw recording (for visualization)
        sorting_gt: Ground truth sorting
    """
    import spikeinterface.full as si
    
    print("\n[1/7] Generating simulated extracellular recording...")
    t0 = time.time()
    
    recording, sorting_gt = si.generate_ground_truth_recording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_units=num_units,
        generate_sorting_kwargs=dict(
            firing_rates=firing_rates,
            refractory_period_ms=refractory_period_ms,
        ),
        noise_kwargs=dict(
            noise_levels=noise_levels,
            strategy='tile_pregenerated',
        ),
        seed=seed,
    )
    
    print(f"  Recording: {recording.get_num_channels()} channels, "
          f"{recording.get_num_samples()} samples, "
          f"{recording.get_sampling_frequency()} Hz")
    print(f"  Duration: {recording.get_total_duration():.1f} s")
    print(f"  GT sorting: {sorting_gt.get_num_units()} units")
    for uid in sorting_gt.unit_ids:
        n_spikes = len(sorting_gt.get_unit_spike_train(uid))
        print(f"    Unit {uid}: {n_spikes} spikes")
    print(f"  Generation took {time.time() - t0:.1f}s")
    
    print("\n[2/7] Preprocessing...")
    t0 = time.time()
    
    recording_f = si.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    recording_cmr = si.common_reference(recording_f, reference='global', operator='median')
    
    print(f"  Bandpass filter: {freq_min}-{freq_max} Hz")
    print(f"  Common median reference applied")
    print(f"  Preprocessing took {time.time() - t0:.1f}s")
    
    return recording_cmr, recording, sorting_gt
