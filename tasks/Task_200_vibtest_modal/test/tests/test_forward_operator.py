import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/vibtest_modal_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    """Main test function for forward_operator."""
    
    # Identify outer and inner data paths
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
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer data func_name: {outer_data.get('func_name')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the forward_operator function
    try:
        print("Executing forward_operator with outer data...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully. Result type: {type(result)}")
    except Exception as e:
        print(f"ERROR: forward_operator execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\nDetected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (an operator/closure)
        if not callable(result):
            print(f"WARNING: Result is not callable but inner data exists. Type: {type(result)}")
            # Fall back to comparing with outer output
            expected = outer_output
        else:
            agent_operator = result
            
            # Load and execute with inner data
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner data func_name: {inner_data.get('func_name')}")
            print(f"Inner args count: {len(inner_args)}")
            
            try:
                print("Executing agent_operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"agent_operator executed successfully. Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: agent_operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\nDetected simple function pattern (no inner data)")
        expected = outer_output
    
    # Phase 3: Verification
    print("\nVerifying results...")
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