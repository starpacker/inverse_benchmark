import torch
import torch.distributed as dist
from ptyrad.utils import vprint, torch_phasor

def evaluate_results(results):
    """
    Finalizes the process, saves logs, cleans up, and calculates evaluation metrics.
    
    Args:
        results (dict): Dictionary containing 'model', 'logger', 'output_path', etc.
        
    Returns:
        dict: A dictionary containing the model, output path, solver time,
              and calculated statistical metrics (shapes, means, stds).
    """
    model = results['model']
    logger = results.get('logger')
    
    vprint("### Evaluation & Cleanup ###")
    
    # --- 1. Cleanup ---
    if logger is not None and hasattr(logger, 'flush_file') and logger.flush_file:
        logger.close()
        
    if dist.is_initialized():
        dist.destroy_process_group()

    # --- 2. Metric Extraction ---
    # Unwrap DDP if present
    inner_model = model.module if hasattr(model, 'module') else model

    def find_tensor(mdl, target_names):
        """
        Robustly finds a tensor (complex or real) matching target names.
        """
        # 1. Check Attributes and Methods directly
        # This catches properties, buffers, and custom objects (like Phasors)
        for name in target_names:
            # Check methods
            if hasattr(mdl, f'get_{name}'):
                return getattr(mdl, f'get_{name}')()
            
            # Check attributes
            for attr in [name, f'_{name}', f'{name}_cmplx']:
                if hasattr(mdl, attr):
                    val = getattr(mdl, attr)
                    # Handle Torch Tensors
                    if isinstance(val, torch.Tensor):
                        return val.detach()
                    # Handle Phasor/Complex wrappers (has .r/.i or .amp/.pha)
                    if hasattr(val, 'r') and hasattr(val, 'i'):
                        return torch.complex(val.r, val.i).detach()
                    if hasattr(val, 'amp') and hasattr(val, 'pha'):
                        return torch.polar(val.amp, val.pha).detach()
                    if hasattr(val, 'real') and hasattr(val, 'imag'):
                        return torch.complex(val.real, val.imag).detach()

        # 2. Check State Dict (for Parameters/Buffers)
        sd = mdl.state_dict()
        keys = list(sd.keys())
        
        for name in target_names:
            # 2a. Direct or Suffix Match (e.g. "obj", "module.obj", "layers.0.obj")
            # We look for keys that end with the name to handle nesting/prefixes
            matches = [k for k in keys if k == name or k.endswith(f'.{name}') or k.endswith(f'_{name}')]
            
            for k in matches:
                val = sd[k]
                if val.is_complex():
                    return val.detach()
                elif val.ndim > 1 and val.shape[-1] == 2:
                    return torch.view_as_complex(val).detach()
                else:
                    # Found a match by name, but it's float. Return it!
                    # This prevents fallback to dummy for real-valued test cases.
                    return val.detach()

            # 2b. Split Parameter Search (.r / .i)
            suffix_pairs = [('.r', '.i'), ('_r', '_i'), ('.real', '.imag'), ('_real', '_imag')]
            for r_suf, i_suf in suffix_pairs:
                # Find keys ending in r_suf where the base name contains target
                r_candidates = [k for k in keys if k.endswith(r_suf) and name in k]
                for r_key in r_candidates:
                    i_key = r_key.replace(r_suf, i_suf)
                    if i_key in sd:
                        return torch.complex(sd[r_key], sd[i_key]).detach()

        return None

    # Attempt to retrieve Object
    obj = find_tensor(inner_model, ['obj', 'object', 'sample', 'psi', 'o'])
    
    # Attempt to retrieve Probe
    probe = find_tensor(inner_model, ['probe', 'illumination', 'p'])

    # Fallback
    if obj is None: 
        vprint("Warning: Could not find object tensor in model. Using dummy.")
        obj = torch.tensor([0.0])
    if probe is None: 
        vprint("Warning: Could not find probe tensor in model. Using dummy.")
        probe = torch.tensor([0.0])

    # Calculate Statistics
    # Ensure we handle real tensors gracefully (abs/angle work on floats too)
    obj_amp = obj.abs()
    obj_phase = obj.angle()

    # --- 3. Construct Output Dictionary ---
    metrics = {
        'output_path': results.get('output_path'),
        'solver_time': results.get('solver_time'),
        'model': model,
        'obj_shape': list(obj.shape),
        'probe_shape': list(probe.shape),
        'obj_amp_mean': obj_amp.mean().item(),
        'obj_amp_std': obj_amp.std().item(),
        'obj_phase_mean': obj_phase.mean().item(),
        'obj_phase_std': obj_phase.std().item(),
    }

    return metrics