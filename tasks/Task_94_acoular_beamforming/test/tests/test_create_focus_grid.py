import sys
import os
import dill
import traceback

# Add path for imports
sys.path.insert(0, '/data/yjh/acoular_beamforming_sandbox_sandbox/run_code')

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agent_create_focus_grid import create_focus_grid
from verification_utils import recursive_check


def main():
    """Main test function for create_focus_grid."""
    
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_create_focus_grid.pkl']
    
    # Classify data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner (parent_function) data file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the outer data file (exact match pattern)
        elif basename == 'standard_data_create_focus_grid.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_focus_grid.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args: {len(outer_args)} arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function with outer arguments
    try:
        result = create_focus_grid(*outer_args, **outer_kwargs)
        print("Successfully called create_focus_grid")
    except Exception as e:
        print(f"ERROR: Failed to execute create_focus_grid: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is a factory pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result from create_focus_grid, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        print(f"Agent operator type: {type(agent_operator)}")
        
        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args: {len(inner_args)} arguments")
            print(f"Inner kwargs: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner arguments
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner data: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\n=== TEST PASSED ===")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        
        expected = outer_output
        actual_result = result
        
        print(f"Expected type: {type(expected)}")
        print(f"Actual result type: {type(actual_result)}")
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            print("Verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\n=== TEST PASSED ===")
        sys.exit(0)


if __name__ == '__main__':
    main()