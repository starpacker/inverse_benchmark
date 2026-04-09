import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to the path to import the target function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_compute_strehl_explicit import compute_strehl_explicit
from verification_utils import recursive_check


def main():
    """Main test function for compute_strehl_explicit."""
    
    # Define data paths
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data_compute_strehl_explicit.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_strehl_explicit.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_strehl_explicit.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("Executing compute_strehl_explicit...")
        result = compute_strehl_explicit(*outer_args, **outer_kwargs)
        
        # Check if this is a factory pattern (result is callable) and we have inner paths
        if callable(result) and inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("Detected factory pattern - result is callable")
            agent_operator = result
            
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                print("Executing agent operator with inner data...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing inner results...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"TEST FAILED (inner execution): {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {os.path.basename(inner_path)}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            # Scenario A: Simple Function
            print("Detected simple function pattern")
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                print(f"Expected type: {type(expected_output)}")
                print(f"Result type: {type(result)}")
                if isinstance(expected_output, (int, float, np.number)):
                    print(f"Expected value: {expected_output}")
                    print(f"Result value: {result}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()