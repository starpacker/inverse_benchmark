import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def forward_operator(weights, pure_components, baselines=None):
    """
    Forward model: Compute mixed spectra from weights and component spectra.
    
    Forward model: mixed_spectrum = sum(w_i * component_i) + baseline
    
    Parameters
    ----------
    weights : ndarray
        (n_mixtures, n_components) array of mixing proportions.
    pure_components : ndarray
        (n_components, n_points) array of pure component spectra.
    baselines : ndarray, optional
        (n_mixtures, n_points) array of baseline contributions.
        If None, no baseline is added.
    
    Returns
    -------
    y_pred : ndarray
        (n_mixtures, n_points) array of predicted mixed spectra.
    """
    # Linear mixing model: Y = W @ H
    y_pred = weights @ pure_components
    
    # Add baseline if provided
    if baselines is not None:
        y_pred = y_pred + baselines
    
    return y_pred
