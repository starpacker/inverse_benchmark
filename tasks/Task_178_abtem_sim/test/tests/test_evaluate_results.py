import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/abtem_sim_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def main():
    """Main test function for evaluate_results"""
    
    # Analyze data paths to determine test strategy
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
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        print("\n[Phase 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("\n[Phase 2] Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function returned type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is a factory pattern (callable result) or simple function
    if len(inner_paths) > 0 and callable(result) and not isinstance(result, (dict, list, np.ndarray)):
        # Scenario B: Factory/Closure Pattern
        print("\n[Phase 3] Detected factory pattern, loading inner data...")
        
        try:
            inner_path = inner_paths[0]  # Use first inner path
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator/closure
            print("\n[Phase 4] Executing the returned operator...")
            actual_result = result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"ERROR in factory pattern execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\n[Phase 3] Simple function pattern detected")
        actual_result = result
        expected = outer_output
    
    # Phase 5: Verification
    try:
        print("\n[Phase 5] Verifying results...")
        print(f"Expected type: {type(expected)}")
        print(f"Actual type: {type(actual_result)}")
        
        passed, msg = recursive_check(expected, actual_result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()