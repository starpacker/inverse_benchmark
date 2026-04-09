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
    data_paths = ['/data/yjh/lenspack_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
        expected_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - args count: {len(outer_args)}")
        print(f"  - kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("\nExecuting evaluate_results with outer args/kwargs...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Check if result is callable (factory pattern) and we have inner data
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure pattern
        print("\nDetected factory pattern. Processing inner data...")
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - args count: {len(inner_args)}")
                print(f"  - kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator/closure
                print("\nExecuting operator with inner args/kwargs...")
                inner_result = result(*inner_args, **inner_kwargs)
                
                # Verify
                print("\nVerifying inner result...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for inner data: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner verification passed: {msg}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - compare result directly
        print("\nSimple function pattern. Verifying result...")
        
        try:
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Verification passed: {msg}")
                print("\nTEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()