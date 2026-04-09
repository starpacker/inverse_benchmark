import sys
import os
import dill
import traceback

# Import the target function
from agent_kappa_from_params import kappa_from_params

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for kappa_from_params."""
    
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_kappa_from_params.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        # Check if this is an inner/parent function data file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_kappa_from_params.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_kappa_from_params.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer data loaded successfully.")
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing kappa_from_params with outer data...")
        result = kappa_from_params(*outer_args, **outer_kwargs)
        print(f"Execution completed. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute kappa_from_params: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # The result should be callable (an operator/closure)
        if not callable(result):
            print(f"WARNING: Expected callable result for factory pattern, got {type(result)}")
            # Fall back to Scenario A comparison
            expected = outer_output
        else:
            # Load inner data and execute the operator
            for inner_path in inner_paths:
                try:
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
                    print("Executing operator with inner data...")
                    result = result(*inner_args, **inner_kwargs)
                    print(f"Operator execution completed. Result type: {type(result)}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to process inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function test")
        expected = outer_output
    
    # Phase 3: Verification
    try:
        print("\nVerifying results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()