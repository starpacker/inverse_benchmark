import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_spiral_array import create_spiral_array
from verification_utils import recursive_check


def main():
    """Main test function for create_spiral_array."""
    
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_create_spiral_array.pkl']
    
    # Phase 1: Identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the exact outer path for create_spiral_array
        elif basename == 'standard_data_create_spiral_array.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_spiral_array.pkl)")
        sys.exit(1)
    
    # Phase 2: Load outer data and reconstruct operator
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
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function with outer args/kwargs
    try:
        agent_result = create_spiral_array(*outer_args, **outer_kwargs)
        print(f"Successfully executed create_spiral_array")
    except Exception as e:
        print(f"ERROR: Failed to execute create_spiral_array: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
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
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner args/kwargs
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"Successfully executed operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # The result from Phase 1 IS the result
        result = agent_result
        expected = outer_output
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                print(f"Expected type: {type(expected)}")
                print(f"Actual type: {type(result)}")
                if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
                    print(f"Expected shape: {expected.shape}")
                    print(f"Actual shape: {result.shape}")
                    print(f"Expected dtype: {expected.dtype}")
                    print(f"Actual dtype: {result.dtype}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()