import sys
import os
import dill
import numpy as np
import traceback

# Add the necessary paths
sys.path.insert(0, '/data/yjh/pylops_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/pylops_sandbox_sandbox/run_code/std_data')

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pylops_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct the function result
    print(f"[Phase 1] Loading outer data from: {outer_path}")
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
    
    print(f"[Phase 1] Outer args: {outer_args}")
    print(f"[Phase 1] Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    print("[Phase 1] Executing load_and_preprocess_data...")
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[Phase 1] Function returned type: {type(agent_result)}")
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[Phase 2] Detected Scenario B (Factory/Closure pattern)")
        print(f"[Phase 2] Found {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"[Phase 2] Loading inner data from: {inner_path}")
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
            
            print(f"[Phase 2] Inner args count: {len(inner_args)}")
            print(f"[Phase 2] Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner arguments
            print("[Phase 2] Executing operator with inner arguments...")
            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("[Phase 2] Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"[Phase 2] Inner test passed for: {os.path.basename(inner_path)}")
    else:
        # Scenario A: Simple function
        print("[Phase 2] Detected Scenario A (Simple function)")
        
        result = agent_result
        expected = outer_output
        
        # Compare results
        print("[Phase 2] Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()