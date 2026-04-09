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
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
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
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer data loaded successfully.")
    print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Args count: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if there are inner paths (factory/closure pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        try:
            # Run the outer function to get the operator/closure
            print("Running evaluate_results to get operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Operator obtained: {type(agent_operator)}")
            
            if not callable(agent_operator):
                print(f"WARNING: Result is not callable, treating as direct result")
                result = agent_operator
                expected = expected_output
            else:
                # Load inner data and execute
                inner_path = inner_paths[0]  # Use first inner path
                print(f"\nLoading inner data from: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner data loaded successfully.")
                print(f"  Function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Args count: {len(inner_args)}")
                print(f"  Kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("\nExecuting operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Result obtained: {type(result)}")
        
        except Exception as e:
            print(f"ERROR: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        
        try:
            print("Running evaluate_results...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            expected = expected_output
            print(f"Result obtained: {type(result)}")
        
        except Exception as e:
            print(f"ERROR: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 2: Verification
    print("\n" + "="*50)
    print("VERIFICATION PHASE")
    print("="*50)
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"\nExpected type: {type(expected)}")
            print(f"Result type: {type(result)}")
            
            # Additional debug info for dictionaries
            if isinstance(expected, dict) and isinstance(result, dict):
                print(f"\nExpected keys: {sorted(expected.keys())}")
                print(f"Result keys: {sorted(result.keys())}")
                
                for key in expected.keys():
                    if key in result:
                        exp_val = expected[key]
                        res_val = result[key]
                        if exp_val != res_val:
                            print(f"\nMismatch in key '{key}':")
                            print(f"  Expected: {exp_val}")
                            print(f"  Got: {res_val}")
            
            sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()