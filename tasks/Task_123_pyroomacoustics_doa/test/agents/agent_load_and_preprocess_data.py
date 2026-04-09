import numpy as np

import matplotlib

matplotlib.use('Agg')

import pyroomacoustics as pra

np.random.seed(42)

def load_and_preprocess_data(
    fs: int,
    room_dim: list,
    c: float,
    n_sources: int,
    n_mics: int,
    mic_radius: float,
    array_center: list,
    signal_duration: float,
    source_positions: list,
    snr_db: float
) -> dict:
    """
    Load and preprocess data: setup room, microphone array, generate source signals,
    simulate room acoustics (forward model), and add sensor noise.
    
    Returns a dictionary containing all necessary data for DOA estimation.
    """
    n_samples = int(fs * signal_duration)
    
    # Compute ground-truth azimuths relative to array center
    true_azimuths_rad = []
    for pos in source_positions:
        dx = pos[0] - array_center[0]
        dy = pos[1] - array_center[1]
        az = np.arctan2(dy, dx)
        if az < 0:
            az += 2 * np.pi
        true_azimuths_rad.append(az)
    true_azimuths_deg = np.degrees(true_azimuths_rad)
    
    print("=" * 60)
    print("Task 123: pyroomacoustics DOA Estimation (MUSIC)")
    print("=" * 60)
    print(f"Room: {room_dim[0]}m x {room_dim[1]}m")
    print(f"Microphone array: {n_mics} mics, circular, radius={mic_radius}m")
    print(f"Number of sources: {n_sources}")
    print(f"True source azimuths (deg): {[f'{a:.1f}' for a in true_azimuths_deg]}")
    print(f"SNR: {snr_db} dB")
    print(f"Sampling rate: {fs} Hz")
    print()
    
    # Step 1: Setup room and microphone array
    print("Step 1: Setting up room and microphone array...")
    
    # Create microphone array (circular, 2D)
    mic_locs = pra.circular_2D_array(array_center, n_mics, 0, mic_radius)
    print(f"  Microphone locations shape: {mic_locs.shape}")
    
    # Create shoebox room (anechoic: max_order=0 for clean DOA)
    room = pra.ShoeBox(room_dim, fs=fs, max_order=0, materials=pra.Material(0.5))
    room.add_microphone_array(mic_locs)
    
    # Step 2: Add sources with distinct signals
    print("Step 2: Adding sound sources...")
    
    source_signals = []
    for i in range(n_sources):
        # Generate broadband signals (white noise)
        sig = np.random.randn(n_samples)
        source_signals.append(sig)
        room.add_source(source_positions[i], signal=sig)
        print(f"  Source {i+1}: position={source_positions[i]}, "
              f"azimuth={true_azimuths_deg[i]:.1f}°")
    
    # Step 3: Forward model - simulate room acoustics
    print("\nStep 3: Simulating room acoustics (forward model)...")
    room.simulate()
    clean_signals = room.mic_array.signals.copy()
    print(f"  Recorded signals shape: {clean_signals.shape}")
    
    # Step 4: Add sensor noise
    print(f"\nStep 4: Adding sensor noise (SNR={snr_db} dB)...")
    signal_power = np.mean(clean_signals ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*clean_signals.shape)
    noisy_signals = clean_signals + noise
    actual_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))
    print(f"  Actual SNR: {actual_snr:.1f} dB")
    
    return {
        "noisy_signals": noisy_signals,
        "clean_signals": clean_signals,
        "mic_locs": mic_locs,
        "true_azimuths_rad": np.array(true_azimuths_rad),
        "true_azimuths_deg": np.array(true_azimuths_deg),
        "source_positions": source_positions,
        "source_signals": source_signals,
        "fs": fs,
        "c": c,
        "n_sources": n_sources,
        "n_mics": n_mics,
        "room_dim": room_dim,
        "mic_radius": mic_radius,
        "array_center": array_center,
        "snr_db": snr_db,
    }
