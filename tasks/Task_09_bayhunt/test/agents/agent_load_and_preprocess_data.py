import os
import numpy as np
import os.path as op
from BayHunter import utils
from BayHunter import SynthObs

def load_and_preprocess_data(data_path='observed', station_idx=3):
    """
    Load and preprocess data for Bayesian inversion.
    Creates config file, generates synthetic test data, and loads observed data.
    
    Returns:
        dict: Dictionary containing all preprocessed data and configuration
    """
    # 1. Create config.ini
    config_content = """[modelpriors]
vpvs = 1.4, 2.1
layers = 1, 20
vs = 2, 5
z = 0, 60
mohoest = None
rfnoise_corr = 0.9
swdnoise_corr = 0.
rfnoise_sigma = 1e-5, 0.05
swdnoise_sigma = 1e-5, 0.05

[initparams]
nchains = 5
iter_burnin = 32768
iter_main = 16384
propdist = 0.015, 0.015, 0.015, 0.005, 0.005
acceptance = 40, 45
thickmin = 0.1
lvz = None
hvz = None
rcond= 1e-5
station = 'test'
savepath = 'results'
maxmodels = 50000
"""
    with open('config.ini', 'w') as f:
        f.write(config_content)
    
    # 2. Create test data directory
    if not op.exists(data_path):
        os.makedirs(data_path)
    
    # 3. Define true model parameters
    h = [5, 23, 8, 0]
    vs = [2.7, 3.6, 3.8, 4.4]
    vpvs = 1.73
    
    # 4. Generate synthetic data
    # Surface waves
    sw_x = np.linspace(1, 41, 21)
    # Note: The '%s' is a placeholder required by BayHunter's save_data for file suffixes
    datafile_sw = op.join(data_path, 'st%d_%s.dat' % (station_idx, '%s'))
    swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)
    SynthObs.save_data(swdata, outfile=datafile_sw)
    
    # Receiver functions
    datafile_rf = op.join(data_path, 'st%d_%s.dat' % (station_idx, '%s'))
    rfdata = SynthObs.return_rfdata(h, vs, vpvs=vpvs, x=None)
    SynthObs.save_data(rfdata, outfile=datafile_rf)
    
    # Save velocity-depth model
    modfile = op.join(data_path, 'st%d_mod.dat' % station_idx)
    SynthObs.save_model(h, vs, vpvs=vpvs, outfile=modfile)
    
    # 5. Load priors and initparams
    initfile = 'config.ini'
    priors, initparams = utils.load_params(initfile)
    
    # 6. Load and Corrupt Data
    # Load the clean synthetic data we just wrote
    xsw, _ysw = np.loadtxt(op.join(data_path, 'st%d_rdispph.dat' % station_idx)).T
    xrf, _yrf = np.loadtxt(op.join(data_path, 'st%d_prf.dat' % station_idx)).T
    
    # Add noise
    noise = [0.0, 0.012, 0.98, 0.005]
    
    # Surface wave noise (Exponential)
    ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise[0], sigma=noise[1])
    ysw = _ysw + ysw_err
    
    # Receiver function noise (Gaussian)
    yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
    yrf = _yrf + yrf_err
    
    # 7. Prepare Reference Model Structure
    dep, vs_mod = np.loadtxt(modfile, usecols=[0, 2], skiprows=1).T
    pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
    pvs = np.repeat(vs_mod, 2)
    
    # 8. Compute Benchmarks
    truenoise = np.concatenate(([noise[0]], [np.std(ysw_err)],
                                [noise[2]], [np.std(yrf_err)]))
    
    explike = SynthObs.compute_explike(yobss=[ysw, yrf], ymods=[_ysw, _yrf],
                                       noise=truenoise, gauss=[False, True],
                                       rcond=initparams['rcond'])
    
    truemodel = {
        'model': (pdep, pvs),
        'nlays': 3,
        'noise': truenoise,
        'explike': explike,
    }
    
    # 9. Update Configuration Objects
    priors.update({
        'mohoest': (38, 4),
        'rfnoise_corr': 0.98,
        'swdnoise_corr': 0.
    })
    
    initparams.update({
        'nchains': 5,
        'iter_burnin': (2048 * 32),
        'iter_main': (2048 * 16),
        'propdist': (0.025, 0.025, 0.015, 0.005, 0.005),
    })
    
    # 10. Return Dictionary
    return {
        'xsw': xsw,
        'ysw': ysw,
        'ysw_err': ysw_err,
        '_ysw': _ysw,
        'xrf': xrf,
        'yrf': yrf,
        'yrf_err': yrf_err,
        '_yrf': _yrf,
        'priors': priors,
        'initparams': initparams,
        'truemodel': truemodel,
        'true_h': h,
        'true_vs': vs,
        'vpvs': vpvs,
        'noise': noise
    }