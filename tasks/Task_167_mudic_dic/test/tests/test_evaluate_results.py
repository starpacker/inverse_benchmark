import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Define data paths
    data_paths = ['/data/yjh/mudic_dic_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Load outer data
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
    
    # Check for inner paths (factory/closure pattern)
    valid_inner_paths = [p for p in inner_paths if os.path.exists(p)]
    
    if len(valid_inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Factory/Closure Pattern (Scenario B)")
        
        try:
            # Phase 1: Create the operator
            print("Phase 1: Creating operator from evaluate_results...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: evaluate_results did not return a callable, got {type(agent_operator)}")
                sys.exit(1)
            
            print(f"Operator created successfully: {type(agent_operator)}")
            
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in valid_inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner data processing: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("Detected Simple Function Pattern (Scenario A)")
        
        try:
            # Execute the function directly
            print("Executing evaluate_results...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            
            expected = outer_output
            
            # Verify results
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()