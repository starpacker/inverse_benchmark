import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_receiver_locations import create_receiver_locations
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_create_receiver_locations.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_create_receiver_locations.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_receiver_locations.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute the target function
    try:
        agent_operator = create_receiver_locations(*outer_args, **outer_kwargs)
        print("Successfully executed create_receiver_locations")
    except Exception as e:
        print(f"ERROR: Failed to execute create_receiver_locations: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result from Phase 1 should be callable
        if not callable(agent_operator):
            print("ERROR: Expected callable operator from factory pattern but got non-callable")
            sys.exit(1)
        
        # Load inner data and execute the operator
        inner_path = inner_paths[0]  # Use the first inner path
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
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed the operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data['output']
    else:
        # Scenario A: Simple function
        # The result from Phase 1 IS the result
        result = agent_operator
        expected = outer_data['output']
    
    # Phase 3: Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()