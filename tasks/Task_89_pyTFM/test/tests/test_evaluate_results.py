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
    data_paths = ['/data/yjh/pyTFM_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data files to determine test strategy
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
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully")
        print(f"  - Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\nExecuting evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nDetected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
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
                inner_expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully")
                print(f"  - Function name: {inner_data.get('func_name', 'unknown')}")
                
                # Execute the operator with inner args
                print("Executing agent operator with inner arguments...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify the result
                print("Verifying inner result...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for inner execution:")
                    print(msg)
                    sys.exit(1)
                
                print(f"Inner verification PASSED")
                
            except Exception as e:
                print(f"ERROR: Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\nDetected simple function pattern")
        
        # Verify the result against expected output
        try:
            print("Verifying result...")
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"VERIFICATION FAILED:")
                print(msg)
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"ERROR: Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()