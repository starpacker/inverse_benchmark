import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute function
    try:
        print("\n=== Phase 2: Executing evaluate_results ===")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Execution completed successfully")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Handle inner data if exists (factory/closure pattern)
    if inner_paths:
        print("\n=== Phase 3: Factory/Closure Pattern Detected ===")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print(f"WARNING: Result is not callable but inner paths exist")
            print("Proceeding with direct comparison of outer result")
            expected = outer_output
        else:
            print(f"Result is callable, loading inner data for execution")
            
            # Load the first inner path
            inner_path = inner_paths[0]
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"Number of inner args: {len(inner_args)}")
                
                # Execute the operator with inner args
                result = result(*inner_args, **inner_kwargs)
                print(f"Inner execution completed successfully")
                
            except Exception as e:
                print(f"ERROR in inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function, compare directly
        print("\n=== Scenario A: Simple Function Pattern ===")
        expected = outer_output
    
    # Phase 4: Verification
    try:
        print("\n=== Phase 4: Verification ===")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print(f"Mismatch details: {msg}")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()