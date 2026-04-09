import sys
import os
import dill
import torch
import numpy as np
import traceback
import math

# Inject math into the gen_std_data module if it exists
try:
    import gen_std_data
    gen_std_data.math = math
except ImportError:
    pass

# Also try to inject into any loaded modules that might need it
import builtins
original_import = builtins.__import__

def custom_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if hasattr(module, '__dict__') and 'math' not in module.__dict__:
        module.__dict__['math'] = math
    return module

# Import the target function
from agent_dps_sample import dps_sample
from verification_utils import recursive_check


def fix_module_math():
    """Ensure math is available in all relevant modules."""
    for module_name, module in list(sys.modules.items()):
        if module is not None and hasattr(module, '__dict__'):
            if 'math' not in module.__dict__:
                try:
                    module.__dict__['math'] = math
                except (TypeError, AttributeError):
                    pass


def load_data(path):
    """Load pickle data with proper error handling."""
    fix_module_math()
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def move_to_device(obj, device):
    """Recursively move tensors to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif hasattr(obj, 'to') and callable(obj.to):
        try:
            return obj.to(device)
        except:
            return obj
    return obj


def main():
    # Data paths provided
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_dps_sample.pkl']
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix math in all modules before loading
    fix_module_math()
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_dps_sample.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_dps_sample.pkl)")
        sys.exit(1)
    
    # Load outer data
    try:
        fix_module_math()
        outer_data = load_data(outer_path)
        print("Loaded outer data successfully.")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Fix math in modules again after loading
    fix_module_math()
    
    # Check if there's a model in args and ensure math is in its module
    if len(outer_args) > 0:
        model = outer_args[0]
        if hasattr(model, '__module__'):
            module_name = model.__module__
            if module_name in sys.modules:
                sys.modules[module_name].__dict__['math'] = math
        
        # Also check the model's class module
        model_class = type(model)
        if hasattr(model_class, '__module__'):
            class_module_name = model_class.__module__
            if class_module_name in sys.modules:
                sys.modules[class_module_name].__dict__['math'] = math
    
    # Inject math into any SinusoidalPositionEmbeddings class
    for module_name, module in list(sys.modules.items()):
        if module is not None:
            try:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name, None)
                    if isinstance(attr, type) and 'Embedding' in attr_name:
                        if hasattr(attr, '__module__'):
                            emb_module = sys.modules.get(attr.__module__)
                            if emb_module:
                                emb_module.__dict__['math'] = math
            except:
                pass
    
    # Scenario check
    if len(inner_paths) > 0:
        # Scenario B: Factory pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        try:
            fix_module_math()
            agent_operator = dps_sample(*outer_args, **outer_kwargs)
            print("Created agent operator successfully.")
        except Exception as e:
            print(f"ERROR creating agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]
        try:
            fix_module_math()
            inner_data = load_data(inner_path)
            print("Loaded inner data successfully.")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            fix_module_math()
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Executed inner operation successfully.")
        except Exception as e:
            print(f"ERROR executing inner operation: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function")
        print("Running dps_sample directly...")
        
        try:
            fix_module_math()
            result = dps_sample(*outer_args, **outer_kwargs)
            print("Executed dps_sample successfully.")
        except Exception as e:
            print(f"ERROR executing dps_sample: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Compare results
    try:
        passed, msg = recursive_check(expected_output, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()