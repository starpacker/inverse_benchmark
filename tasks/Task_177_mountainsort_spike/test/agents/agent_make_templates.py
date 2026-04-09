import numpy as np

import matplotlib

matplotlib.use("Agg")

def make_templates(num_units, template_samples, num_channels):
    """Create distinct waveform templates across channels."""
    t = np.linspace(0, 1, template_samples, endpoint=False)
    templates = np.zeros((num_units, template_samples, num_channels))

    # Unit 0 – biphasic, large on ch0-ch1
    wave0 = -np.sin(2 * np.pi * t) * np.exp(-3 * t)
    templates[0, :, 0] = wave0 * 1.0
    templates[0, :, 1] = wave0 * 0.7
    templates[0, :, 2] = wave0 * 0.15
    templates[0, :, 3] = wave0 * 0.10

    # Unit 1 – triphasic, large on ch2-ch3
    wave1 = (np.sin(3 * np.pi * t) * np.exp(-4 * t))
    templates[1, :, 0] = wave1 * 0.10
    templates[1, :, 1] = wave1 * 0.15
    templates[1, :, 2] = wave1 * 1.0
    templates[1, :, 3] = wave1 * 0.8

    # Unit 2 – monophasic negative, spread across channels
    wave2 = -np.exp(-((t - 0.25) ** 2) / (2 * 0.04 ** 2))
    templates[2, :, 0] = wave2 * 0.5
    templates[2, :, 1] = wave2 * 0.3
    templates[2, :, 2] = wave2 * 0.4
    templates[2, :, 3] = wave2 * 0.9

    return templates
