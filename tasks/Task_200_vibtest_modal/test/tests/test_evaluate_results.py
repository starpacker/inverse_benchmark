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
    
    # Data paths provided
    data_paths = ['/data/yjh/vibtest_modal_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  Function name: {outer_data.get('func_name')}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\nExecuting evaluate_results with outer args/kwargs...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nDetected factory pattern with {len(inner_paths)} inner data file(s).")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  Function name: {inner_data.get('func_name')}")
                
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("\nExecuting agent_operator with inner args/kwargs...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                print("\nVerifying results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("\nDetected simple function pattern (no inner data files).")
        
        expected = outer_output
        actual_result = result
        
        # Verify results
        try:
            print("\nVerifying results...")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("Verification passed.")
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()