import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_make_schedule import make_schedule

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for make_schedule."""
    
    # Data paths provided
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_make_schedule.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_schedule.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_make_schedule.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute make_schedule with outer args/kwargs
    try:
        agent_result = make_schedule(*outer_args, **outer_kwargs)
        print("Successfully called make_schedule with outer args/kwargs")
    except Exception as e:
        print(f"ERROR: Failed to execute make_schedule: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Detected Scenario B: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"WARNING: agent_result is not callable (type: {type(agent_result)}), treating as Scenario A")
            # Fall back to Scenario A
            result = agent_result
            expected = outer_output
        else:
            # Load inner data and execute
            inner_path = inner_paths[0]  # Use first inner path
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
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner args/kwargs
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner args/kwargs")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function (no inner data files)")
        result = agent_result
        expected = outer_output
    
    # Phase 3: Verification
    print("\n--- Verification Phase ---")
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(result)}")
    
    if expected is None:
        print("WARNING: Expected output is None, cannot verify")
        sys.exit(1)
    
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
        
        # Additional debug info for dictionaries
        if isinstance(expected, dict) and isinstance(result, dict):
            print("\n--- Debug Info ---")
            print(f"Expected keys: {list(expected.keys())}")
            print(f"Result keys: {list(result.keys())}")
            for key in expected.keys():
                if key in result:
                    exp_val = expected[key]
                    res_val = result[key]
                    if isinstance(exp_val, torch.Tensor) and isinstance(res_val, torch.Tensor):
                        print(f"Key '{key}': expected shape {exp_val.shape}, result shape {res_val.shape}")
                        if exp_val.shape == res_val.shape:
                            diff = torch.abs(exp_val - res_val).max().item()
                            print(f"  Max absolute difference: {diff}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()