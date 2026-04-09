import numpy as np
from BayHunter import SynthObs

def forward_operator(model_params, data_type='swd', x=None):
    """
    Forward operator that computes synthetic data from model parameters.
    
    Args:
        model_params: dict with 'h' (layer thicknesses), 'vs' (shear velocities), 'vpvs' (vp/vs ratio)
        data_type: 'swd' for surface wave dispersion, 'rf' for receiver function
        x: x-axis values (periods for SWD, time for RF)
    
    Returns:
        numpy array: Predicted data
    """
    h = model_params['h']
    vs = model_params['vs']
    vpvs = model_params.get('vpvs', 1.73)
    
    if data_type == 'swd':
        if x is None:
            x = np.linspace(1, 41, 21)
        # Return surface wave dispersion data
        swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=x)
        # swdata is a dict with keys like 'rdispph', 'rdispgr', etc.
        # Extract the Rayleigh phase dispersion values
        if 'rdispph' in swdata:
            return swdata['rdispph'][1]  # Return y values (phase velocities)
        else:
            # Return the first available dispersion curve
            for key in swdata:
                if swdata[key] is not None:
                    return swdata[key][1]
        return None
    
    elif data_type == 'rf':
        # Return receiver function data
        rfdata = SynthObs.return_rfdata(h, vs, vpvs=vpvs, x=x)
        # rfdata is a dict with keys like 'prf', 'srf'
        if 'prf' in rfdata:
            return rfdata['prf'][1]  # Return y values (RF amplitudes)
        else:
            for key in rfdata:
                if rfdata[key] is not None:
                    return rfdata[key][1]
        return None
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")