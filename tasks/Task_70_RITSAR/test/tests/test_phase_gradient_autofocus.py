import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_phase_gradient_autofocus import phase_gradient_autofocus

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for phase_gradient_autofocus."""
    
    # Data paths provided
    data_paths = ['/data/yjh/RITSAR_sandbox_sandbox/run_code/std_data/standard_data_phase_gradient_autofocus.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_phase_gradient_autofocus.pkl':
            outer_path = path
    
    # Verify we have at least the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_phase_gradient_autofocus.pkl")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
    except Exception as e:
        print(f"ERROR: Failed to extract data from outer_data")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on presence of inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Run the function to get the operator/closure
        try:
            agent_operator = phase_gradient_autofocus(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to create operator from phase_gradient_autofocus")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Extract inner args, kwargs, and expected output
            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
            except Exception as e:
                print(f"ERROR: Failed to extract data from inner_data")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner arguments")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner path: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Run the function
        try:
            result = phase_gradient_autofocus(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute phase_gradient_autofocus")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Expected output
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()