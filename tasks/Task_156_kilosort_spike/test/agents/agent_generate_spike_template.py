import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_spike_template(n_samples=61, template_type=0):
    """Generate realistic biphasic spike waveform templates."""
    t = np.linspace(-1, 2, n_samples)
    if template_type == 0:
        template = -np.exp(-t**2/0.1) + 0.3*np.exp(-(t-0.5)**2/0.2)
    elif template_type == 1:
        template = -0.8*np.exp(-t**2/0.08) + 0.5*np.exp(-(t-0.4)**2/0.15)
    elif template_type == 2:
        template = -1.2*np.exp(-t**2/0.12) + 0.2*np.exp(-(t-0.6)**2/0.25)
    else:
        template = -0.6*np.exp(-t**2/0.06) + 0.4*np.exp(-(t-0.3)**2/0.1)
    return template / np.abs(template).max()
