import sys
import os
import dill
import numpy as np
import traceback

# Add the script directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/sigpy_mri_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Execute forward_operator to get the operator/closure
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: forward_operator did not return a callable, got {type(agent_operator)}")
            sys.exit(1)
        
        print("Successfully created operator from forward_operator")
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner data function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Number of inner args: {len(inner_args)}")
            print(f"Number of inner kwargs: {len(inner_kwargs)}")
            
            # Execute the operator with inner data
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        # Execute forward_operator directly
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print(f"TEST PASSED: {msg}")
        sys.exit(0)


if __name__ == "__main__":
    main()