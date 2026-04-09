import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_ground_truth import create_ground_truth

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for create_ground_truth."""
    
    # Data paths provided
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_create_ground_truth.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_create_ground_truth.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_ground_truth.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run create_ground_truth
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
    
    # Execute the target function
    try:
        agent_result = create_ground_truth(*outer_args, **outer_kwargs)
        print("Successfully executed create_ground_truth")
    except Exception as e:
        print(f"ERROR: Failed to execute create_ground_truth: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable result for factory pattern but got non-callable")
            sys.exit(1)
        
        agent_operator = agent_result
        
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
            expected_output = inner_data.get('output', None)
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare direct output
        expected_output = outer_output
        actual_result = agent_result
        
        try:
            passed, msg = recursive_check(expected_output, actual_result)
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