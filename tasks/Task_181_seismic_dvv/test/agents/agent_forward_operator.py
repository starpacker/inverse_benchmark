import numpy as np

from scipy.interpolate import interp1d

def forward_operator(ccf_ref: np.ndarray, t: np.ndarray, dvv: float) -> np.ndarray:
    """
    Forward operator: apply velocity change to reference CCF.
    
    CCF_cur(t) = CCF_ref(t * (1 + eps)) where eps = -dv/v
    
    Parameters:
        ccf_ref: reference cross-correlation function
        t: time axis array
        dvv: velocity change (dv/v)
    
    Returns:
        ccf_cur: stretched/perturbed CCF
    """
    eps = -dvv
    t_stretched = t * (1.0 + eps)
    interp_func = interp1d(t, ccf_ref, kind='cubic',
                           bounds_error=False, fill_value=0.0)
    return interp_func(t_stretched)
