import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_compute_ssim import compute_ssim

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for compute_ssim."""
    
    # Define data paths
    data_paths = ['/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_compute_ssim.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_ssim.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_ssim.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - args count: {len(outer_args)}")
        print(f"  - kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing compute_ssim with outer args/kwargs...")
        agent_result = compute_ssim(*outer_args, **outer_kwargs)
        print(f"Function executed successfully.")
        print(f"  - Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR executing compute_ssim: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nDetected factory pattern with {len(inner_paths)} inner data file(s).")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - args count: {len(inner_args)}")
                print(f"  - kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing agent operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully.")
                print(f"  - Result type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR executing agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                print("\nVerifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner path: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function - result is the output
        print("\nDetected simple function pattern (no inner data files).")
        
        result = agent_result
        expected = outer_output
        
        # Verify results
        try:
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()