import numbers

import numpy as np

def parse_harmonic(harmonic, harmonic_max=None):
    """Parses harmonic input into a list of integers."""
    if harmonic_max is not None and harmonic_max < 1:
        raise ValueError(f'{harmonic_max=} < 1')

    if harmonic is None:
        return [1], False

    if isinstance(harmonic, (int, numbers.Integral)):
        if harmonic < 1 or (harmonic_max is not None and harmonic > harmonic_max):
            raise IndexError(f'{harmonic=!r} is out of bounds [1, {harmonic_max}]')
        return [int(harmonic)], False

    if isinstance(harmonic, str):
        if harmonic == 'all':
            if harmonic_max is None:
                raise TypeError(f'maximum harmonic must be specified for {harmonic=!r}')
            return list(range(1, harmonic_max + 1)), True
        raise ValueError(f'invalid {harmonic=!r}')

    h = np.atleast_1d(harmonic)
    if h.size == 0:
        raise ValueError(f'{harmonic=!r} is empty')
    return [int(i) for i in harmonic], True

def forward_operator(signal, axis=0, harmonic=1):
    """
    Computes the Forward Model: Signal (Time Domain) -> Phasor (Frequency Domain).
    Returns (mean, real, imag) components.
    """
    signal = np.asarray(signal)
    
    # Handle negative indexing or dimension matching
    ndim = signal.ndim
    if axis < 0:
        axis += ndim
        
    samples = signal.shape[axis]
    harmonic_list, has_harmonic_axis = parse_harmonic(harmonic, samples // 2)
    
    # Prepare for FFT: move signal axis to last
    if axis != ndim - 1:
        signal_swapped = np.swapaxes(signal, axis, -1)
    else:
        signal_swapped = signal

    # Real FFT
    # signal_swapped shape: (..., samples)
    fft_values = np.fft.rfft(signal_swapped, axis=-1)
    
    # Extract DC (0th component)
    dc = fft_values[..., 0].real
    
    # Avoid division by zero
    valid_dc = np.abs(dc) > 1e-9
    
    means = dc / samples
    
    reals = []
    imags = []
    
    for h in harmonic_list:
        if h < fft_values.shape[-1]:
            val = fft_values[..., h]
            
            # Normalization logic:
            # Phasor G (real) = Re(DFT) / DC
            # Phasor S (imag) = -Im(DFT) / DC  (Note the sign flip for standard phasor definition)
            
            r = np.zeros_like(dc)
            i = np.zeros_like(dc)
            
            r[valid_dc] = val.real[valid_dc] / dc[valid_dc]
            i[valid_dc] = -val.imag[valid_dc] / dc[valid_dc]
            
            reals.append(r)
            imags.append(i)
        else:
            raise ValueError(f"Harmonic {h} too high for samples {samples}")
            
    # Return formatted arrays
    if len(harmonic_list) == 1 and not has_harmonic_axis:
        return means, reals[0], imags[0]
    else:
        return means, np.stack(reals, axis=0), np.stack(imags, axis=0)
