import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/GPRPy_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Scenario A: Simple function (only outer data exists)
    if outer_path and not inner_paths:
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Function name: {outer_data.get('func_name')}")
            
            # Execute the function
            result = evaluate_results(*outer_args, **outer_kwargs)
            
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error during test execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (both outer and inner data exist)
    elif outer_path and inner_paths:
        try:
            # Phase 1: Load outer data and create operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            
            # Create the operator/closure
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            # Verify it's callable
            if not callable(agent_operator):
                print(f"TEST FAILED: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            # Phase 2: Load inner data and execute
            for inner_path in inner_paths:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Loaded inner data from: {inner_path}")
                
                # Execute the operator
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected_output, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Error during test execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print("TEST FAILED: No valid data files found")
        sys.exit(1)

if __name__ == "__main__":
    main()