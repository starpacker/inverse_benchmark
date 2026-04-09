import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path to import the target module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_shrink import shrink
from verification_utils import recursive_check


def main():
    """Main test function for shrink."""
    
    # Data paths provided (empty in this case)
    data_paths = []
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if path.endswith('_shrink.pkl') and 'parent_function' not in path:
            outer_path = path
        elif 'parent_function' in path and '_shrink_' in path:
            inner_paths.append(path)
    
    # If no data paths provided, we need to handle this case
    if not data_paths:
        # No data files provided - create a simple test case
        print("No data files provided. Running basic functionality test...")
        
        try:
            # Test with simple inputs based on the function signature
            # shrink(x, L) -> returns xs = sign(x) * max(abs(x) - 1/L, 0)
            x_test = np.array([1.0, -2.0, 0.5, -0.3, 3.0])
            L_test = 2.0
            
            result = shrink(x_test, L_test)
            
            # Manually compute expected result
            s = np.abs(x_test)
            expected = np.sign(x_test) * np.maximum(s - 1 / L_test, 0)
            
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"TEST FAILED: Exception during basic test: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario A or B based on file structure
    try:
        if outer_path is None:
            print("TEST FAILED: No outer data file found")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run the function with outer args
        agent_result = shrink(*outer_args, **outer_kwargs)
        
        # Phase 2: Check if we have inner paths (Scenario B) or not (Scenario A)
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
            
            # Verify agent_result is callable
            if not callable(agent_result):
                print(f"TEST FAILED: Expected callable operator, got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                # Execute the operator with inner args
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"Passed: {inner_path}")
            
            print("TEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple Function
            print("Detected simple function pattern")
            
            result = agent_result
            expected = outer_data.get('output')
            
            # Compare results
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"TEST FAILED: Data file not found: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    except Exception as e:
        print(f"TEST FAILED: Unexpected exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()