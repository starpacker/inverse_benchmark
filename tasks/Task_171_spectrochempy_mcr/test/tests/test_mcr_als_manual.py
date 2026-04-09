import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_mcr_als_manual import mcr_als_manual

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for mcr_als_manual."""
    
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_mcr_als_manual.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_mcr_als_manual.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_mcr_als_manual.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and run mcr_als_manual
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing mcr_als_manual with outer args/kwargs...")
        result = mcr_als_manual(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute mcr_als_manual: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    valid_inner_paths = [p for p in inner_paths if os.path.exists(p)]
    
    if valid_inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B detected: Found {len(valid_inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner data exists. Proceeding with direct comparison.")
            # Fall back to Scenario A behavior
            expected = outer_output
        else:
            # Process inner data
            inner_path = valid_inner_paths[0]  # Use first inner path
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args/kwargs...")
                result = result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR: Failed in inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: No inner data files found")
        expected = outer_output
    
    # Phase 3: Verification
    try:
        print("Comparing results...")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        # Additional debug info for tuples
        if isinstance(expected, tuple) and isinstance(result, tuple):
            print(f"Expected tuple length: {len(expected)}")
            print(f"Result tuple length: {len(result)}")
            for i, (e, r) in enumerate(zip(expected, result)):
                print(f"  Element {i}: expected type={type(e)}, result type={type(r)}")
                if hasattr(e, 'shape'):
                    print(f"    Expected shape: {e.shape}")
                if hasattr(r, 'shape'):
                    print(f"    Result shape: {r.shape}")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()