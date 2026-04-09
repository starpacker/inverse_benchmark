import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner path (contains parent_function pattern)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer path (exact match pattern)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the forward_operator function
    try:
        print("Executing forward_operator with outer data...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Execution completed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, but inner data exists.")
            print("Falling back to direct comparison with outer output.")
            expected = outer_output
        else:
            # Process inner paths
            for inner_path in inner_paths:
                try:
                    print(f"\nLoading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output')
                    
                    print(f"Inner data loaded successfully.")
                    print(f"  - Function name: {inner_data.get('func_name', 'unknown')}")
                    
                    # Execute the operator with inner args
                    print("Executing operator with inner data...")
                    result = result(*inner_args, **inner_kwargs)
                    expected = inner_output
                    print("Inner execution completed successfully.")
                    
                except Exception as e:
                    print(f"ERROR: Failed to process inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function execution")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("\nVerifying results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print("\n--- Expected ---")
            if isinstance(expected, dict):
                for k, v in list(expected.items())[:5]:
                    print(f"  {k}: {v}")
                if len(expected) > 5:
                    print(f"  ... and {len(expected) - 5} more items")
            else:
                print(f"  Type: {type(expected)}")
                print(f"  Value: {expected}")
            
            print("\n--- Actual ---")
            if isinstance(result, dict):
                for k, v in list(result.items())[:5]:
                    print(f"  {k}: {v}")
                if len(result) > 5:
                    print(f"  ... and {len(result) - 5} more items")
            else:
                print(f"  Type: {type(result)}")
                print(f"  Value: {result}")
            
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()