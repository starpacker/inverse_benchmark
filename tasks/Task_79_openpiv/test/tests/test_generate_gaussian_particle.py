import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_gaussian_particle import generate_gaussian_particle

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for generate_gaussian_particle."""
    
    # Data paths provided
    data_paths = ['/data/yjh/openpiv_sandbox_sandbox/run_code/std_data/standard_data_generate_gaussian_particle.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_gaussian_particle.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_gaussian_particle.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = generate_gaussian_particle(*outer_args, **outer_kwargs)
        print("Successfully executed generate_gaussian_particle")
    except Exception as e:
        print(f"ERROR: Failed to execute generate_gaussian_particle: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Process inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with outer output
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()