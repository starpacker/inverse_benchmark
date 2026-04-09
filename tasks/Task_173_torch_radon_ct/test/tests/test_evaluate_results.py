import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the module to the path
sys.path.insert(0, '/data/yjh/torch_radon_ct_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Test evaluate_results function against standard data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/torch_radon_ct_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Identify outer path (main function data) and inner path (closure/operator data)
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        print("Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if result is callable (factory pattern) and we have inner data
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure pattern
        print("Detected factory/closure pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data for: {inner_data.get('func_name', 'unknown')}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner function executed successfully")
            except Exception as e:
                print(f"ERROR executing inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - result is already computed
        print("Simple function pattern (Scenario A)")
    
    # Phase 3: Verification
    try:
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, result)
        
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