import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_solve_darcy_flow_fast import solve_darcy_flow_fast

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for solve_darcy_flow_fast."""
    
    # Data paths provided
    data_paths = ['/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_solve_darcy_flow_fast.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_solve_darcy_flow_fast.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_solve_darcy_flow_fast.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
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
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing solve_darcy_flow_fast with outer args/kwargs...")
        agent_result = solve_darcy_flow_fast(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute solve_darcy_flow_fast: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory pattern. Found {len(inner_paths)} inner data file(s).")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator but got {type(agent_result)}")
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
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args/kwargs...")
                actual_result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner test: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern (no inner data files).")
        
        # Compare the result with expected output
        try:
            print("Comparing results...")
            passed, msg = recursive_check(outer_output, agent_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()