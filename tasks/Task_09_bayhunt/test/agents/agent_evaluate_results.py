import os.path as op
from BayHunter import PlotFromStorage

def evaluate_results(results_dict, maxmodels=100000, dev=0.05):
    """
    Evaluate and save results from the inversion.
    
    Args:
        results_dict: Dictionary from run_inversion containing metadata like
                      'savepath', 'station', and 'truemodel'.
        maxmodels: Maximum number of models to save in the final ensemble.
        dev: Deviation threshold (percentage) for outlier chain detection.
    
    Returns:
        dict: Evaluation metrics and summary including paths and status.
    """
    # Extract metadata from the results dictionary
    savepath = results_dict['savepath']
    station = results_dict['station']
    truemodel = results_dict['truemodel']
    
    # Construct the path to the configuration pickle file
    cfile = '%s_config.pkl' % station
    configfile = op.join(savepath, 'data', cfile)
    
    # Initialize the plotting object with the configuration file
    obj = PlotFromStorage(configfile)
    
    # Prune outlier chains based on 'dev' and save the top 'maxmodels'
    obj.save_final_distribution(maxmodels=maxmodels, dev=dev)
    
    # Generate and save visual plots (velocity profiles, likelihoods)
    obj.save_plots(refmodel=truemodel)
    
    # Compile a summary dictionary for the user
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