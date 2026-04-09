import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def find_test_files(data_paths):
    """
    Categorize data files into outer (main function) and inner (closure/operator) files.
    """
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def load_pickle_file(filepath):
    """
    Load a pickle file using dill.
    """
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"ERROR: Failed to load pickle file '{filepath}': {e}")
        traceback.print_exc()
        return None


def main():
    # Define data paths
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Categorize files
    outer_path, inner_paths = find_test_files(data_paths)
    
    if outer_path is None:
        print("ERROR: No outer data file (standard_data_forward_operator.pkl) found.")
        sys.exit(1)
    
    print(f"Outer data file: {outer_path}")
    print(f"Inner data files: {inner_paths}")
    
    # Phase 1: Load outer data and execute forward_operator
    print("\n=== Phase 1: Loading outer data and executing forward_operator ===")
    
    outer_data = load_pickle_file(outer_path)
    if outer_data is None:
        print("ERROR: Failed to load outer data.")
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    try:
        # Execute the forward_operator function
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully.")
        print(f"Result type: {type(result)}")
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    print("\n=== Phase 2: Verification ===")
    
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Scenario B detected: Factory/Closure Pattern")
        
        # Check if result is callable (an operator/closure)
        if callable(result):
            agent_operator = result
            print("Result is callable, treating as operator/closure.")
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                inner_data = load_pickle_file(inner_path)
                if inner_data is None:
                    print(f"ERROR: Failed to load inner data from '{inner_path}'.")
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    print(f"Operator executed successfully.")
                except Exception as e:
                    print(f"ERROR: Failed to execute operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify results
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner data verification PASSED.")
            
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            # Result is not callable, fall back to Scenario A
            print("Result is not callable, falling back to Scenario A logic.")
            expected = outer_output
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: Simple Function")
        expected = outer_output
    
    # Verify results for Scenario A (or Scenario B fallback)
    if expected is None:
        print("WARNING: No expected output found in outer data.")
    
    passed, msg = recursive_check(expected, result)
    if not passed:
        print(f"VERIFICATION FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()