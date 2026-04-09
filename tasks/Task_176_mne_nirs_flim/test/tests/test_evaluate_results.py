import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/mne_nirs_flim_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def main():
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    # Check if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Run the outer function to get the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from outer function call")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify the result
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")
        
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully executed evaluate_results")
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Verify the result
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()