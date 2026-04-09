import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_csm_nnls_inversion import csm_nnls_inversion
from verification_utils import recursive_check

def main():
    """Main test function for csm_nnls_inversion."""
    
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_csm_nnls_inversion.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        # Check if this is an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_csm_nnls_inversion.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_csm_nnls_inversion.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing csm_nnls_inversion with outer args/kwargs...")
        result = csm_nnls_inversion(*outer_args, **outer_kwargs)
        print(f"Execution completed. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern) and we have inner data
    if callable(result) and not isinstance(result, np.ndarray) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern. Loading inner data...")
        
        try:
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            print("Executing operator with inner args/kwargs...")
            actual_result = result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"ERROR during inner execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Simple function pattern detected.")
        actual_result = result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Running verification...")
        passed, msg = recursive_check(expected, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()