import sys
import os
import dill
import traceback

# Import numpy if available
try:
    import numpy as np
except ImportError:
    np = None

# Import the target function
from agent_electron_wavelength_angstrom_local import electron_wavelength_angstrom_local
from verification_utils import recursive_check

def main():
    """Main test function for electron_wavelength_angstrom_local."""
    
    # Data paths provided
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_electron_wavelength_angstrom_local.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_electron_wavelength_angstrom_local.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_electron_wavelength_angstrom_local.pkl)")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    # Phase 1: Call the function with outer args
    try:
        result = electron_wavelength_angstrom_local(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute electron_wavelength_angstrom_local: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()