import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_rng_generators(expected, actual):
    """
    Compare two numpy random Generator objects by comparing their bit_generator state.
    """
    if not (hasattr(expected, 'bit_generator') and hasattr(actual, 'bit_generator')):
        return False, "One or both objects don't have bit_generator attribute"
    
    try:
        expected_state = expected.bit_generator.state
        actual_state = actual.bit_generator.state
        
        # Compare the state dictionaries
        if expected_state['bit_generator'] != actual_state['bit_generator']:
            return False, f"bit_generator type mismatch: {expected_state['bit_generator']} vs {actual_state['bit_generator']}"
        
        # Compare state values
        exp_state_val = expected_state['state']
        act_state_val = actual_state['state']
        
        if set(exp_state_val.keys()) != set(act_state_val.keys()):
            return False, "State keys mismatch"
        
        for key in exp_state_val:
            if isinstance(exp_state_val[key], np.ndarray):
                if not np.array_equal(exp_state_val[key], act_state_val[key]):
                    return False, f"State array mismatch at key '{key}'"
            else:
                if exp_state_val[key] != act_state_val[key]:
                    return False, f"State value mismatch at key '{key}': {exp_state_val[key]} vs {act_state_val[key]}"
        
        return True, "RNG generators match"
    except Exception as e:
        return False, f"Error comparing RNG generators: {e}"


def custom_recursive_check(expected, actual, path="output"):
    """
    Custom comparison that handles numpy random Generator objects specially.
    """
    # Check for numpy random Generator
    if isinstance(expected, np.random.Generator) and isinstance(actual, np.random.Generator):
        return compare_rng_generators(expected, actual)
    
    # Handle tuples and lists
    if isinstance(expected, (tuple, list)):
        if type(expected) != type(actual):
            return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(exp_item, act_item, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "All elements match"
    
    # Handle dicts
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch at {path}: expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}"
        
        for key in expected:
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}[{key}]")
            if not passed:
                return False, msg
        return True, "All dict items match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch at {path}: expected ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if expected.dtype != actual.dtype:
            return False, f"Dtype mismatch at {path}: expected {expected.dtype}, got {actual.dtype}"
        
        # Handle complex arrays
        if np.iscomplexobj(expected) or np.iscomplexobj(actual):
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, f"Array values mismatch at {path}"
        else:
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
                return False, f"Array values mismatch at {path}"
        return True, "Arrays match"
    
    # For other types, use recursive_check
    return recursive_check(expected, actual)


def main():
    # Define data paths
    data_paths = ['/data/yjh/CDI_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Outer data loaded successfully.")
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args: {outer_args}")
        print(f"  Kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected Scenario B: Factory/Closure pattern with {len(inner_paths)} inner data file(s)")
        
        if not callable(agent_result):
            print(f"ERROR: Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Inner data loaded successfully.")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"  Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Inner args count: {len(inner_args)}")
                print(f"  Inner kwargs: {inner_kwargs}")
                
                print("Executing operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
                print("Comparing results...")
                passed, msg = custom_recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function pattern
        print("Detected Scenario A: Simple function pattern")
        
        expected = outer_data.get('output')
        result = agent_result
        
        print("Comparing results...")
        passed, msg = custom_recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()