import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_plot_results import plot_results
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_plot_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_plot_results.pkl':
            outer_path = path
    
    # Check if outer_path exists
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_plot_results.pkl)")
        sys.exit(1)
    
    # Load outer data
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
    expected_output = outer_data.get('output')
    
    # Scenario B: Check if inner paths exist (factory/closure pattern)
    if inner_paths:
        # Phase 1: Reconstruct Operator
        try:
            agent_operator = plot_results(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully called plot_results to get operator")
        except Exception as e:
            print(f"ERROR: Failed to create operator from plot_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print("ERROR: Returned operator is not callable")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
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
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple function (no inner paths)
        try:
            result = plot_results(*outer_args, **outer_kwargs)
            print("Successfully called plot_results")
        except Exception as e:
            print(f"ERROR: Failed to execute plot_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()