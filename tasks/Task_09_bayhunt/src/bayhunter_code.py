import os
import numpy as np
import os.path as op
import matplotlib
matplotlib.use('PDF')
import logging

# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from BayHunter import PlotFromStorage
from BayHunter import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer
from BayHunter import ModelMatrix
from BayHunter import SynthObs


def load_and_preprocess_data(data_path='observed', station_idx=3):
    """
    Load and preprocess data for Bayesian inversion.
    Creates config file, generates synthetic test data, and loads observed data.
    
    Returns:
        dict: Dictionary containing all preprocessed data and configuration
    """
    # Create config.ini
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
iter_burnin = (2048 * 16)
iter_main = (2048 * 8)
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
    
    # Create test data directory
    if not op.exists(data_path):
        os.makedirs(data_path)
    
    # Define true model parameters
    h = [5, 23, 8, 0]
    vs = [2.7, 3.6, 3.8, 4.4]
    vpvs = 1.73
    
    # Generate surface waves data
    sw_x = np.linspace(1, 41, 21)
    datafile = op.join(data_path, 'st%d_%s.dat' % (station_idx, '%s'))
    swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)
    SynthObs.save_data(swdata, outfile=datafile)
    
    # Generate receiver functions data
    pars = {'p': 6.4}
    datafile = op.join(data_path, 'st%d_%s.dat' % (station_idx, '%s'))
    rfdata = SynthObs.return_rfdata(h, vs, vpvs=vpvs, x=None)
    SynthObs.save_data(rfdata, outfile=datafile)
    
    # Save velocity-depth model
    modfile = op.join(data_path, 'st%d_mod.dat' % station_idx)
    SynthObs.save_model(h, vs, vpvs=vpvs, outfile=modfile)
    
    # Load priors and initparams from config.ini
    initfile = 'config.ini'
    priors, initparams = utils.load_params(initfile)
    
    # Load observed data
    xsw, _ysw = np.loadtxt(op.join(data_path, 'st%d_rdispph.dat' % station_idx)).T
    xrf, _yrf = np.loadtxt(op.join(data_path, 'st%d_prf.dat' % station_idx)).T
    
    # Add noise to create observed data
    noise = [0.0, 0.012, 0.98, 0.005]
    ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise[0], sigma=noise[1])
    ysw = _ysw + ysw_err
    yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
    yrf = _yrf + yrf_err
    
    # Load reference model
    dep, vs_mod = np.loadtxt(modfile, usecols=[0, 2], skiprows=1).T
    pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
    pvs = np.repeat(vs_mod, 2)
    
    # Compute true noise and expected likelihood
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
    
    # Update priors
    priors.update({
        'mohoest': (38, 4),
        'rfnoise_corr': 0.98,
        'swdnoise_corr': 0.
    })
    
    # Update initparams
    initparams.update({
        'nchains': 5,
        'iter_burnin': (2048 * 32),
        'iter_main': (2048 * 16),
        'propdist': (0.025, 0.025, 0.015, 0.005, 0.005),
    })
    
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


def run_inversion(data_dict, nthreads=6, baywatch=True, dtsend=1):
    """
    Run the MCMC Bayesian inversion.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data
        nthreads: Number of threads for parallel inversion
        baywatch: Whether to enable BayWatch monitoring
        dtsend: Time interval for sending data to BayWatch
    
    Returns:
        dict: Results including optimizer and path information
    """
    xsw = data_dict['xsw']
    ysw = data_dict['ysw']
    ysw_err = data_dict['ysw_err']
    xrf = data_dict['xrf']
    yrf = data_dict['yrf']
    priors = data_dict['priors']
    initparams = data_dict['initparams']
    truemodel = data_dict['truemodel']
    
    # Define targets
    target1 = Targets.RayleighDispersionPhase(xsw, ysw, yerr=ysw_err)
    target2 = Targets.PReceiverFunction(xrf, yrf)
    target2.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)
    
    # Join targets
    targets = Targets.JointTarget(targets=[target1, target2])
    
    # Save config for baywatch
    utils.save_baywatch_config(targets, path='.', priors=priors,
                               initparams=initparams, refmodel=truemodel)
    
    # Create optimizer
    optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                               random_seed=None)
    
    # Run inversion
    optimizer.mp_inversion(nthreads=nthreads, baywatch=baywatch, dtsend=dtsend)
    
    return {
        'optimizer': optimizer,
        'savepath': initparams['savepath'],
        'station': initparams['station'],
        'targets': targets,
        'truemodel': truemodel
    }


def evaluate_results(results_dict, maxmodels=100000, dev=0.05):
    """
    Evaluate and save results from the inversion.
    
    Args:
        results_dict: Dictionary from run_inversion
        maxmodels: Maximum number of models to save
        dev: Deviation threshold for outlier detection
    
    Returns:
        dict: Evaluation metrics and summary
    """
    savepath = results_dict['savepath']
    station = results_dict['station']
    truemodel = results_dict['truemodel']
    
    # Load results from storage
    cfile = '%s_config.pkl' % station
    configfile = op.join(savepath, 'data', cfile)
    
    obj = PlotFromStorage(configfile)
    
    # Save final distribution (excludes outlier chains)
    obj.save_final_distribution(maxmodels=maxmodels, dev=dev)
    
    # Save plots with reference model
    obj.save_plots(refmodel=truemodel)
    
    # Compute evaluation metrics
    evaluation = {
        'configfile': configfile,
        'savepath': savepath,
        'maxmodels': maxmodels,
        'dev': dev,
        'truemodel_nlays': truemodel['nlays'],
        'truemodel_explike': truemodel['explike'],
        'status': 'completed'
    }
    
    return evaluation


if __name__ == '__main__':
    # Setup logging
    formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
    logging.basicConfig(format=formatter, level=logging.INFO)
    logger = logging.getLogger()
    
    # Step 1: Load and preprocess data
    print("=" * 60)
    print("Step 1: Loading and preprocessing data...")
    print("=" * 60)
    data_dict = load_and_preprocess_data(data_path='observed', station_idx=3)
    print(f"Loaded SWD data: x shape = {data_dict['xsw'].shape}, y shape = {data_dict['ysw'].shape}")
    print(f"Loaded RF data: x shape = {data_dict['xrf'].shape}, y shape = {data_dict['yrf'].shape}")
    
    # Step 2: Test forward operator
    print("=" * 60)
    print("Step 2: Testing forward operator...")
    print("=" * 60)
    model_params = {
        'h': data_dict['true_h'],
        'vs': data_dict['true_vs'],
        'vpvs': data_dict['vpvs']
    }
    
    # Test SWD forward model
    y_pred_swd = forward_operator(model_params, data_type='swd', x=data_dict['xsw'])
    if y_pred_swd is not None:
        print(f"Forward model SWD output shape: {y_pred_swd.shape}")
        print(f"SWD prediction range: [{y_pred_swd.min():.4f}, {y_pred_swd.max():.4f}]")
    else:
        print("SWD forward model returned None")
    
    # Test RF forward model
    y_pred_rf = forward_operator(model_params, data_type='rf', x=data_dict['xrf'])
    if y_pred_rf is not None:
        print(f"Forward model RF output shape: {y_pred_rf.shape}")
        print(f"RF prediction range: [{y_pred_rf.min():.4f}, {y_pred_rf.max():.4f}]")
    else:
        print("RF forward model returned None")
    
    # Step 3: Run inversion
    print("=" * 60)
    print("Step 3: Running MCMC Bayesian inversion...")
    print("=" * 60)
    results_dict = run_inversion(data_dict, nthreads=6, baywatch=True, dtsend=1)
    print("Inversion completed.")
    
    # Step 4: Evaluate results
    print("=" * 60)
    print("Step 4: Evaluating results...")
    print("=" * 60)
    evaluation = evaluate_results(results_dict, maxmodels=100000, dev=0.05)
    print(f"Results saved to: {evaluation['savepath']}")
    print(f"Status: {evaluation['status']}")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")