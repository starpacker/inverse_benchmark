import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def build_step(depth_list, values):
    """Build step-function arrays for plotting."""
    d_plot = [0]
    v_plot = [values[0]]
    for i, d in enumerate(depth_list):
        d_plot.append(d)
        v_plot.append(values[i])
        d_plot.append(d)
        v_plot.append(values[i+1] if i+1 < len(values) else values[-1])
    d_plot.append(depth_list[-1] + 500)
    v_plot.append(values[-1])
    return d_plot, v_plot
