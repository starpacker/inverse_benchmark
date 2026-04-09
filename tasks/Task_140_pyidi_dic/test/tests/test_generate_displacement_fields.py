import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_displacement_fields import generate_displacement_fields

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for generate_displacement_fields."""
    
    # Define data paths
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data_generate_displacement_fields.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_displacement_fields.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_displacement_fields.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    try:
        result = generate_displacement_fields(*outer_args, **outer_kwargs)
        print("Successfully executed generate_displacement_fields")
    except Exception as e:
        print(f"ERROR: Failed to execute generate_displacement_fields: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) and we have inner data
    if callable(result) and not isinstance(result, (np.ndarray, tuple, list)) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected factory pattern - result is callable")
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Successfully loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output', None)
        
        # Execute the operator
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function - result is the output
        print("Detected simple function pattern")
    
    # Phase 2: Verification
    if expected_output is None:
        print("WARNING: No expected output found in data file")
        sys.exit(1)
    
    try:
        passed, msg = recursive_check(expected_output, result)
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