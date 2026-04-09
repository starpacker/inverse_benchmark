import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_visualize_projections_internal import visualize_projections_internal
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_visualize_projections_internal.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize_projections_internal.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize_projections_internal.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
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
    
    # Check if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            print("Executing visualize_projections_internal to get operator...")
            agent_operator = visualize_projections_internal(*outer_args, **outer_kwargs)
            print(f"Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Agent operator is not callable, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
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
            
            try:
                print("Executing agent operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        try:
            print("Executing visualize_projections_internal...")
            result = visualize_projections_internal(*outer_args, **outer_kwargs)
            print(f"Execution completed. Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Comparison
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()