import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_fourier_design_matrix import fourier_design_matrix
from verification_utils import recursive_check

def main():
    """Main test function for fourier_design_matrix."""
    
    # Data paths provided
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data_fourier_design_matrix.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_fourier_design_matrix.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_fourier_design_matrix.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct/run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Run the function with outer args
    try:
        agent_result = fourier_design_matrix(*outer_args, **outer_kwargs)
        print("Successfully executed fourier_design_matrix with outer args")
    except Exception as e:
        print(f"ERROR: Failed to execute fourier_design_matrix: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: The result is a callable, we need to execute it with inner args
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        if not callable(agent_result):
            print(f"WARNING: Expected callable from fourier_design_matrix, got {type(agent_result)}")
            # Fall back to Scenario A comparison
            print("Falling back to Scenario A comparison...")
            result = agent_result
            expected = outer_output
        else:
            # Load inner data and execute
            inner_path = inner_paths[0]  # Use first inner path
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed the callable with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute callable with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            expected = inner_output
    else:
        # Scenario A: Simple function, compare directly
        print("Scenario A detected: Simple function comparison")
        result = agent_result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()