import sys
import os
import dill
import numpy as np
import traceback

# Add the path to allow imports
sys.path.insert(0, '/data/yjh/sasview_saxs_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/sasview_saxs_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        print("\n[Phase 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("\n[Phase 2] Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully.")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Check if result is callable (factory pattern)
    if callable(result) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n[Phase 3] Detected factory pattern, loading inner data...")
        
        try:
            inner_path = inner_paths[0]  # Use first inner path
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)
            
            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the returned callable
            print("\n[Phase 4] Executing inner function...")
            result = result(*inner_args, **inner_kwargs)
            print(f"Inner function executed successfully.")
            
        except Exception as e:
            print(f"ERROR in factory pattern execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function - result is already the output
        print("\n[Phase 3] Simple function pattern detected.")
    
    # Phase 4: Verification
    try:
        print("\n[Verification] Comparing results...")
        print(f"Expected type: {type(expected_output)}")
        print(f"Actual type: {type(result)}")
        
        if isinstance(expected_output, dict):
            print(f"Expected keys: {list(expected_output.keys())}")
        if isinstance(result, dict):
            print(f"Actual keys: {list(result.keys())}")
        
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("\n" + "="*60)
            print("TEST PASSED")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("TEST FAILED")
            print(f"Mismatch details: {msg}")
            print("="*60)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()