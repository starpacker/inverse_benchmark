import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/suncasa_radio_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Categorize paths: outer (main function) vs inner (parent_function pattern)
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl in data_paths")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
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
    
    # Determine scenario based on inner paths
    inner_path = None
    for ip in inner_paths:
        if os.path.exists(ip):
            inner_path = ip
            break
    
    if inner_path is not None:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Factory/Closure pattern")
        print(f"Inner data path: {inner_path}")
        
        # Phase 1: Reconstruct the operator
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from forward_operator")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Load inner data
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Successfully loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        # Phase 2: Execute the operator with inner args
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Scenario A: Simple Function
        print(f"\nScenario A detected: Simple function call")
        
        # Execute forward_operator directly
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print("Successfully executed forward_operator")
        except Exception as e:
            print(f"ERROR: Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
    
    # Comparison phase
    print("\n--- Verification Phase ---")
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: recursive_check raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        
        # Additional debug info
        print("\n--- Debug Info ---")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        if isinstance(expected, tuple) and isinstance(result, tuple):
            print(f"Expected tuple length: {len(expected)}")
            print(f"Result tuple length: {len(result)}")
            for i, (e, r) in enumerate(zip(expected, result)):
                print(f"  Element {i}: expected type={type(e)}, result type={type(r)}")
                if hasattr(e, 'shape') and hasattr(r, 'shape'):
                    print(f"    Expected shape: {e.shape}, Result shape: {r.shape}")
                if hasattr(e, 'dtype') and hasattr(r, 'dtype'):
                    print(f"    Expected dtype: {e.dtype}, Result dtype: {r.dtype}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()