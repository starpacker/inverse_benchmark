import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_load_and_preprocess_data import load_and_preprocess_data
    import agent_load_and_preprocess_data as agent_module
except ImportError:
    print("Error: Could not import 'agent_load_and_preprocess_data'. Ensure the file exists.")
    sys.exit(1)

from verification_utils import recursive_check

def relaxed_recursive_check(expected, actual, path=""):
    """
    A wrapper around recursive_check that handles the specific namespace mismatch 
    issue (e.g., __main__.Field vs agent_module.Field) by relaxing type checks 
    for custom classes if their data matches.
    """
    # 1. Handle NumPy arrays (including subclasses like Field)
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: {expected.shape} vs {actual.shape}"
        
        # Check strict equality for small integers/booleans, close equality for floats
        if np.issubdtype(expected.dtype, np.number):
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
                return False, f"Value mismatch at {path}"
        else:
            if not np.array_equal(expected, actual):
                return False, f"Content mismatch at {path}"
        
        # Check attributes if they are custom subclasses (like Field having a .grid)
        # We only check attributes present in 'expected'
        if hasattr(expected, '__dict__'):
            for k, v in expected.__dict__.items():
                if k.startswith('_'): continue
                if not hasattr(actual, k):
                    return False, f"Missing attribute {k} at {path}"
                act_v = getattr(actual, k)
                passed, msg = relaxed_recursive_check(v, act_v, path=f"{path}.{k}")
                if not passed:
                    return False, msg
        return True, ""

    # 2. Handle Dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}: {set(expected.keys())} vs {set(actual.keys())}"
        for k in expected:
            passed, msg = relaxed_recursive_check(expected[k], actual[k], path=f"{path}['{k}']")
            if not passed: return False, msg
        return True, ""

    # 3. Handle Lists/Tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: {len(expected)} vs {len(actual)}"
        for i, (e_item, a_item) in enumerate(zip(expected, actual)):
            passed, msg = relaxed_recursive_check(e_item, a_item, path=f"{path}[{i}]")
            if not passed: return False, msg
        return True, ""

    # 4. Handle Custom Objects (The Namespace Fix)
    # If types don't strictly match but names match (e.g. Field vs Field), check attributes
    type_e = type(expected)
    type_a = type(actual)
    
    if type_e != type_a:
        name_e = type_e.__name__
        name_a = type_a.__name__
        
        # Allow mismatch if class names are the same (e.g. Field)
        if name_e == name_a:
            # Check public attributes
            if hasattr(expected, '__dict__'):
                e_vars = {k: v for k, v in vars(expected).items() if not k.startswith('_')}
                a_vars = {k: v for k, v in vars(actual).items() if not k.startswith('_')}
                
                # Check keys
                if set(e_vars.keys()) != set(a_vars.keys()):
                    return False, f"Attribute keys mismatch for relaxed type {name_e} at {path}"
                
                for k in e_vars:
                    passed, msg = relaxed_recursive_check(e_vars[k], a_vars[k], path=f"{path}.{k}")
                    if not passed: return False, msg
                return True, ""
            else:
                # If no dict, try string representation or equality
                if str(expected) == str(actual):
                    return True, ""

    # Fallback to standard recursive check for primitives or strict matches
    return recursive_check(expected, actual)

def run_test():
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify File Roles
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if "parent_function" in p:
            inner_path = p
        elif "load_and_preprocess_data.pkl" in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_load_and_preprocess_data.pkl not found.")
        sys.exit(1)

    # 2. Load Data
    # To fix the namespace issue during dill.load, we can try to map __main__ classes to the agent module
    # or just rely on relaxed checking. Let's rely on relaxed checking first.
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 3. Execute
    try:
        print("Executing load_and_preprocess_data...")
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # This function is usually parameterless, but we support args if they exist
        result = load_and_preprocess_data(*args, **kwargs)
        
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify
    print("Verifying results...")
    
    # Determine expected output
    if inner_path:
        # Scenario B (Closure/Factory): Not applicable for this specific function signature,
        # but kept for architectural consistency.
        pass
    else:
        # Scenario A: Direct comparison
        expected = outer_data['output']

    # Use relaxed check to handle 'Field' class mismatch
    passed, msg = relaxed_recursive_check(expected, result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()