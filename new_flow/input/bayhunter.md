# main.py
```python
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

# ------------------------------------------------------------
# Helper: Create config.ini
# ------------------------------------------------------------
def create_config_file():
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
    print("Created config.ini")

# ------------------------------------------------------------
# Helper: Create test data
# ------------------------------------------------------------
def create_test_data():
    if not op.exists('observed'):
        os.makedirs('observed')
        
    idx = 3
    h = [5, 23, 8, 0]
    vs = [2.7, 3.6, 3.8, 4.4]
    vpvs = 1.73
    path = 'observed'

    # surface waves
    sw_x = np.linspace(1, 41, 21)
    datafile = op.join(path, 'st%d_%s.dat' % (idx, '%s'))
    swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)
    SynthObs.save_data(swdata, outfile=datafile)

    # receiver functions
    pars = {'p': 6.4}
    # datafile is reused/reset but logic is similar
    datafile = op.join(path, 'st%d_%s.dat' % (idx, '%s'))
    rfdata = SynthObs.return_rfdata(h, vs, vpvs=vpvs, x=None)
    SynthObs.save_data(rfdata, outfile=datafile)

    # velocity-depth model
    modfile = op.join(path, 'st%d_mod.dat' % idx)
    SynthObs.save_model(h, vs, vpvs=vpvs, outfile=modfile)
    print("Created test data in observed/")

# ------------------------------------------------------------
# Main Logic (from tutorialhunt.py)
# ------------------------------------------------------------
def main():
    #
    # console printout formatting
    #
    formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
    logging.basicConfig(format=formatter, level=logging.INFO)
    logger = logging.getLogger()

    #
    # ------------------------------------------------------------  obs SYNTH DATA
    #
    # Load priors and initparams from config.ini or simply create dictionaries.
    initfile = 'config.ini'
    priors, initparams = utils.load_params(initfile)

    # Load observed data (synthetic test data)
    # The tutorial script hardcodes 'observed/st3_rdispph.dat'
    # Ensure create_test_data created these specific files.
    # create_test_data uses idx=3, so st3_rdispph.dat should exist.
    xsw, _ysw = np.loadtxt('observed/st3_rdispph.dat').T
    xrf, _yrf = np.loadtxt('observed/st3_prf.dat').T

    # add noise to create observed data
    # order of noise values (correlation, amplitude):
    # noise = [corr1, sigma1, corr2, sigma2] for 2 targets
    noise = [0.0, 0.012, 0.98, 0.005]
    ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise[0], sigma=noise[1])
    ysw = _ysw + ysw_err
    yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
    yrf = _yrf + yrf_err

    #
    # -------------------------------------------  get reference model for BayWatch
    #
    # Create truemodel only if you wish to have reference values in plots
    # and BayWatch. You ONLY need to assign the values in truemodel that you
    # wish to have visible.
    dep, vs = np.loadtxt('observed/st3_mod.dat', usecols=[0, 2], skiprows=1).T
    pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
    pvs = np.repeat(vs, 2)

    truenoise = np.concatenate(([noise[0]], [np.std(ysw_err)],   # target 1
                                [noise[2]], [np.std(yrf_err)]))  # target 2

    explike = SynthObs.compute_explike(yobss=[ysw, yrf], ymods=[_ysw, _yrf],
                                       noise=truenoise, gauss=[False, True],
                                       rcond=initparams['rcond'])
    truemodel = {'model': (pdep, pvs),
                 'nlays': 3,
                 'noise': truenoise,
                 'explike': explike,
                 }

    #
    #  -----------------------------------------------------------  DEFINE TARGETS
    #
    target1 = Targets.RayleighDispersionPhase(xsw, ysw, yerr=ysw_err)
    target2 = Targets.PReceiverFunction(xrf, yrf)
    target2.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)

    # Join the targets. targets must be a list instance with all targets
    # you want to use for MCMC Bayesian inversion.
    targets = Targets.JointTarget(targets=[target1, target2])

    #
    #  ---------------------------------------------------  Quick parameter update
    #
    priors.update({'mohoest': (38, 4),  # optional, moho estimate (mean, std)
                   'rfnoise_corr': 0.98,
                   'swdnoise_corr': 0.
                   # 'rfnoise_sigma': np.std(yrf_err),  # fixed to true value
                   # 'swdnoise_sigma': np.std(ysw_err),  # fixed to true value
                   })

    initparams.update({'nchains': 5,
                       'iter_burnin': (2048 * 32),
                       'iter_main': (2048 * 16),
                       'propdist': (0.025, 0.025, 0.015, 0.005, 0.005),
                       })

    #
    #  -------------------------------------------------------  MCMC BAY INVERSION
    #
    # Save configfile for baywatch. refmodel must not be defined.
    utils.save_baywatch_config(targets, path='.', priors=priors,
                               initparams=initparams, refmodel=truemodel)
    optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                               random_seed=None)
    # default for the number of threads is the amount of cpus == one chain per cpu.
    # if baywatch is True, inversion data is continuously send out (dtsend)
    # to be received by BayWatch (see below).
    optimizer.mp_inversion(nthreads=6, baywatch=True, dtsend=1)

    #
    # #  ---------------------------------------------- Model resaving and plotting
    path = initparams['savepath']
    cfile = '%s_config.pkl' % initparams['station']
    configfile = op.join(path, 'data', cfile)
    obj = PlotFromStorage(configfile)
    # The final distributions will be saved with save_final_distribution.
    # Beforehand, outlier chains will be detected and excluded.
    # Outlier chains are defined as chains with a likelihood deviation
    # of dev * 100 % from the median posterior likelihood of the best chain.
    obj.save_final_distribution(maxmodels=100000, dev=0.05)
    # Save a selection of important plots
    obj.save_plots(refmodel=truemodel)
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")

if __name__ == "__main__":
    create_config_file()
    create_test_data()
    main()
```
