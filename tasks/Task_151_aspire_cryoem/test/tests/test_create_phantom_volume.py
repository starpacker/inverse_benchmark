import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_phantom_volume import create_phantom_volume

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_create_phantom_volume.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_create_phantom_volume.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_phantom_volume.pkl)")
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
    
    try:
        agent_operator = create_phantom_volume(*outer_args, **outer_kwargs)
        print("Successfully called create_phantom_volume with outer args/kwargs")
    except Exception as e:
        print(f"ERROR: Failed to execute create_phantom_volume: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_path is not None and os.path.exists(inner_path):
        # Scenario B: Factory/Closure pattern
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
        
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable for Scenario B")
            sys.exit(1)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner args/kwargs")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_operator
        expected = outer_data.get('output')
        print("Scenario A: Simple function - using outer data output for comparison")
    
    # Comparison
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