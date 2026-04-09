import matplotlib

matplotlib.use('Agg')

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

EPS_HBO_760 = 1486.5865

EPS_HBR_760 = 3843.707

EPS_HBO_850 = 2526.391

EPS_HBR_850 = 1798.643

DPF_760 = 6.0

DPF_850 = 5.5

D = 3.0

def forward_operator(hbo, hbr):
    """
    Forward Modified Beer-Lambert Law.
    ΔOD(λ) = ε_HbO(λ)·Δ[HbO]·DPF(λ)·d + ε_HbR(λ)·Δ[HbR]·DPF(λ)·d
    
    Parameters
    ----------
    hbo : ndarray
        HbO concentration changes (in Molar)
    hbr : ndarray
        HbR concentration changes (in Molar)
        
    Returns
    -------
    od_760 : ndarray
        Optical density change at 760nm
    od_850 : ndarray
        Optical density change at 850nm
    """
    od_760 = (EPS_HBO_760 * hbo * DPF_760 * D +
              EPS_HBR_760 * hbr * DPF_760 * D)
    od_850 = (EPS_HBO_850 * hbo * DPF_850 * D +
              EPS_HBR_850 * hbr * DPF_850 * D)
    return od_760, od_850
