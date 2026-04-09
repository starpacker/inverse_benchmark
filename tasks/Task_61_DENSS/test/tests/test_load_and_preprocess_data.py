import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/DENSS_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Verify outer_path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
    print(f"[TEST] Loading outer data from: {outer_path}")
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
    expected_output = outer_data.get('output', None)
    
    print(f"[TEST] Outer args: {outer_args}")
    print(f"[TEST] Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    print("[TEST] Executing load_and_preprocess_data...")
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable, and we need to test inner execution
        print(f"[TEST] Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        if not callable(agent_result):
            print(f"ERROR: Expected callable from load_and_preprocess_data, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"[TEST] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"[TEST] Inner args: {inner_args}")
            print(f"[TEST] Inner kwargs: {inner_kwargs}")
            
            # Execute the operator
            print("[TEST] Executing operator with inner args...")
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("[TEST] Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: Comparison failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"[TEST] Inner test passed for: {inner_path}")
    else:
        # Scenario A: Simple function, compare result directly
        print("[TEST] Detected simple function pattern (Scenario A)")
        print("[TEST] Comparing results...")
        
        try:
            passed, msg = recursive_check(expected_output, agent_result)
        except Exception as e:
            print(f"ERROR: Comparison failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()