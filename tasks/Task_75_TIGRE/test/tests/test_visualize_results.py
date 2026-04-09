import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_visualize_results import visualize_results
from verification_utils import recursive_check

def main():
    """Main test function for visualize_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_visualize_results.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
    expected_output = outer_data.get('output')
    
    print(f"Function name: {outer_data.get('func_name')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        try:
            # Execute outer function to get the operator/closure
            agent_operator = visualize_results(*outer_args, **outer_kwargs)
            print("Successfully created agent operator from outer function call")
        except Exception as e:
            print(f"ERROR: Failed to execute outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute the operator
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
        expected_output = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        try:
            result = visualize_results(*outer_args, **outer_kwargs)
            print("Successfully executed visualize_results function")
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 3: Compare results
    try:
        passed, msg = recursive_check(expected_output, result)
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