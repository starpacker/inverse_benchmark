import numpy as np
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Detector import Detector
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
import matplotlib.pyplot as plt

def get_slopes_diffractive_explicit(wfs, phase_in=None):
    """
    Simulates the physical process of the Shack-Hartmann WFS:
    1. Propagate phase to lenslet array.
    2. Form spots (PSFs) for each subaperture via FFT.
    3. Compute Center of Gravity (CoG) of spots to get slopes.
    """
    if phase_in is not None:
        wfs.telescope.src.phase = phase_in

    cube_em = wfs.get_lenslet_em_field(wfs.telescope.src.phase)
    complex_field = np.fft.fft2(cube_em, axes=[1, 2])
    intensity_spots = np.abs(complex_field)**2
    
    n_pix = intensity_spots.shape[1]
    x = np.arange(n_pix) - n_pix // 2
    X, Y = np.meshgrid(x, x)
    
    slopes = np.zeros((wfs.nValidSubaperture, 2))
    valid_idx = 0
    
    for i in range(wfs.nSubap**2):
        if wfs.valid_subapertures_1D[i]:
            I = intensity_spots[i]
            flux = np.sum(I)
            
            if flux > 0:
                cx = np.sum(I * X) / flux
                cy = np.sum(I * Y) / flux
                slopes[valid_idx, 0] = cx
                slopes[valid_idx, 1] = cy
                valid_idx += 1
                
    slopes_flat = np.concatenate((slopes[:, 0], slopes[:, 1]))
    
    return slopes_flat

def forward_operator(data, dm_commands=None):
    """
    Forward model: Given DM actuator commands, compute the resulting WFS slopes.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing system components. 
        Supports keys: ['telescope'/'tel', 'source'/'ngs', 'atmosphere'/'atm', 
                        'deformable_mirror'/'dm', 'wfs', 'ref_slopes']
    dm_commands : np.ndarray or None
        Actuator commands for the DM. If None, assumes zero commands (flat DM).
        
    Returns:
    --------
    slopes : np.ndarray
        Measured WFS slopes (residual after subtracting reference).
    """
    tel = data.get('telescope', data.get('tel'))
    ngs = data.get('source', data.get('ngs'))
    atm = data.get('atmosphere', data.get('atm'))
    dm  = data.get('deformable_mirror', data.get('dm'))
    wfs = data['wfs']
    ref_slopes = data['ref_slopes']
    
    if tel is None: raise KeyError("Telescope object not found in data dict (checked 'telescope' and 'tel')")
    if dm is None: raise KeyError("DM object not found in data dict (checked 'deformable_mirror' and 'dm')")

    if dm_commands is None:
        dm.coefs = np.zeros(dm.nValidAct)
    else:
        dm.coefs = dm_commands.copy()
    
    atm * ngs * tel * dm
    
    slopes_meas = get_slopes_diffractive_explicit(wfs)
    slopes_residual = slopes_meas - ref_slopes
    
    return slopes_residual