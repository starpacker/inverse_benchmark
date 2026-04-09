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
