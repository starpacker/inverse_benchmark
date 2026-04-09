import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_compute_relative_l2 import compute_relative_l2

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for compute_relative_l2."""
    
    # Define data paths
    data_paths = ['/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_compute_relative_l2.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_relative_l2.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_compute_relative_l2.pkl'")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the target function
        print("Executing compute_relative_l2 with outer arguments...")
        agent_result = compute_relative_l2(*outer_args, **outer_kwargs)
        
        # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print("Detected Scenario B: Factory/Closure pattern")
            
            # Verify the result is callable
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator but got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            # Process each inner path
            for inner_path in inner_paths:
                if not os.path.exists(inner_path):
                    print(f"WARNING: Inner data file does not exist: {inner_path}")
                    continue
                
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner arguments
                print("Executing agent_operator with inner arguments...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for inner path: {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            
            print("TEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple function
            print("Detected Scenario A: Simple function")
            
            result = agent_result
            expected = outer_output
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Failure message: {msg}")
                print(f"Expected type: {type(expected)}")
                print(f"Result type: {type(result)}")
                if isinstance(expected, (int, float, np.number)):
                    print(f"Expected value: {expected}")
                    print(f"Result value: {result}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
    
    except Exception as e:
        print(f"ERROR: Exception during test execution")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()