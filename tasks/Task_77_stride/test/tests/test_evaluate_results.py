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
    data_paths = ['/data/yjh/stride_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern) and we have inner data
    if callable(result) and not isinstance(result, (dict, np.ndarray)) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected factory pattern - result is callable")
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        # Execute the operator
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Operator executed successfully")
        except Exception as e:
            print(f"ERROR executing operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function - result is already the output
        print("Simple function pattern - using direct result")
    
    # Comparison
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()