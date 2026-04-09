from BayHunter import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer

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
    # 1. Unpack Data
    xsw = data_dict['xsw']
    ysw = data_dict['ysw']
    ysw_err = data_dict['ysw_err']
    xrf = data_dict['xrf']
    yrf = data_dict['yrf']
    priors = data_dict['priors']
    initparams = data_dict['initparams']
    truemodel = data_dict.get('truemodel', None)
    
    # 2. Define targets
    # Target 1: Surface Wave Dispersion
    target1 = Targets.RayleighDispersionPhase(xsw, ysw, yerr=ysw_err)
    
    # Target 2: Receiver Function
    target2 = Targets.PReceiverFunction(xrf, yrf)
    # Set specific physical parameters for the RF simulation
    target2.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)
    
    # 3. Join targets
    targets = Targets.JointTarget(targets=[target1, target2])
    
    # 4. Save config for baywatch (Real-time monitoring)
    utils.save_baywatch_config(targets, path='.', priors=priors,
                               initparams=initparams, refmodel=truemodel)
    
    # 5. Create optimizer
    optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                               random_seed=None)
    
    # 6. Run inversion (Multi-processing)
    optimizer.mp_inversion(nthreads=nthreads, baywatch=baywatch, dtsend=dtsend)
    
    return {
        'optimizer': optimizer,
        'savepath': initparams['savepath'],
        'station': initparams['station'],
        'targets': targets,
        'truemodel': truemodel
    }