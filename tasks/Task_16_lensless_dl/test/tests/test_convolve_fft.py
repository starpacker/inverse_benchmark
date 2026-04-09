import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_convolve_fft import convolve_fft
from verification_utils import recursive_check


def main():
    """Main test function for convolve_fft."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_convolve_fft.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_convolve_fft.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_convolve_fft.pkl")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_expected = outer_data.get('output', None)
    
    # Check if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # The outer function returns an operator/closure
        try:
            agent_operator = convolve_fft(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute convolve_fft with outer args")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: convolve_fft did not return a callable, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute operator
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare result with expected
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner path: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - direct execution and comparison
        try:
            result = convolve_fft(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute convolve_fft")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected
        try:
            passed, msg = recursive_check(outer_expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()