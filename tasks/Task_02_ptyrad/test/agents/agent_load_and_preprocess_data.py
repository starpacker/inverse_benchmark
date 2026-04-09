import os
import yaml
import torch

# --- REQUIRED IMPORTS ---
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint,
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.params import PtyRADParams

def _normalize_constraint_params(constraint_params):
    """
    Helper to convert old constraint param format into a standardized
    start/step/end iteration format.
    """
    normalized_params = {}
    for name, p in constraint_params.items():
        # Extract scheduling logic
        freq = p.get("freq", None)
        
        # Logic: If freq is present, start at 1, otherwise use provided start_iter
        start_iter = p.get("start_iter", 1 if freq is not None else None)
        
        # Logic: If freq is present, use it as step, otherwise default to 1
        step = p.get("step", freq if freq is not None else 1)
        
        end_iter = p.get("end_iter", None)
        
        # Reconstruct dict with standardized keys, removing the old ones
        normalized_params[name] = {
            "start_iter": start_iter,
            "step": step,
            "end_iter": end_iter,
            # Include all other keys (e.g., 'alpha', 'threshold') that aren't scheduling keys
            **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
        }
    return normalized_params

def load_and_preprocess_data(params_path, gpuid=0, logger=None):
    """
    Loads configuration, initializes system, and prepares initial data structures.
    
    Args:
        params_path (str): Path to the .yaml configuration file.
        gpuid (int): GPU index to use.
        logger (CustomLogger, optional): Existing logger instance. If None, creates new one.
        
    Returns:
        dict: A context dictionary containing params, device, logger, 
              init (Initializer), variables, loss_fn, constraint_fn, and verbose flag.
    """
    # 1. System Setup
    print_system_info()
    
    # Initialize logger only if not provided to prevent overwriting existing logs
    if logger is None:
        logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    
    device = set_gpu_device(gpuid=gpuid)

    # 2. Load Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")
    
    vprint(f"### Loading params file: {params_path} ###")
    with open(params_path, "r", encoding='utf-8') as file:
        raw_params = yaml.safe_load(file)
    
    # Normalize constraints (legacy support)
    if raw_params.get('constraint_params') is not None:
        raw_params['constraint_params'] = _normalize_constraint_params(raw_params['constraint_params'])
    
    # Validate params using the Pydantic model
    params = PtyRADParams(**raw_params).model_dump()
    params['params_path'] = params_path

    # 3. Initialize Data (Measurements, Probe, Object guess)
    vprint("### Initializing Initializer ###")
    # init_all() triggers the actual data loading/generation
    initializer = Initializer(params['init_params'], seed=None).init_all()
    init_variables = initializer.init_variables

    # 4. Initialize Loss and Constraints
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=True)

    # Pack everything into a context dictionary
    data_context = {
        "params": params,
        "device": device,
        "logger": logger,
        "init": initializer,            # FIX: Key must be 'init'
        "init_variables": init_variables,
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn,
        "accelerator": None,
        "verbose": params.get('verbose', True) # FIX: Key 'verbose' is required by test harness
    }
    
    return data_context