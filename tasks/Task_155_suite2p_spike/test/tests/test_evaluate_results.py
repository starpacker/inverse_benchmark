import sys
import os
import dill
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Test script for evaluate_results function."""
    
    data_paths = ['/data/yjh/suite2p_spike_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner data paths
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
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        print("Executing evaluate_results with outer data...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # The result should be callable (an operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, treating as Scenario A instead")
            expected = outer_output
        else:
            # Load inner data and execute the operator
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                print("Executing operator with inner data...")
                result = result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function call")
        expected = outer_output
    
    # Phase 2: Verification
    print("\nVerifying results...")
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()