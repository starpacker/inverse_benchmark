import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_sky_model import create_sky_model
from verification_utils import recursive_check

def main():
    """Main test function for create_sky_model."""
    
    # Data paths provided
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_create_sky_model.pkl']
    
    # Step 1: Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_create_sky_model.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_sky_model.pkl)")
        sys.exit(1)
    
    # Step 2: Load outer data
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
    
    # Step 3: Phase 1 - Execute the function to reconstruct operator/result
    try:
        agent_result = create_sky_model(*outer_args, **outer_kwargs)
        print(f"Successfully executed create_sky_model")
        print(f"Result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute create_sky_model: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(agent_result)}")
            sys.exit(1)
        
        # Load inner data and execute
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
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner args
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Successfully executed operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        result = agent_result
        expected = outer_output
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()