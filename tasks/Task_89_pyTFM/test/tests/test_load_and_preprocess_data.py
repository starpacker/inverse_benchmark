import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pyTFM_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze file paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Scenario A: Simple function test (only outer data exists)
    if outer_path and not inner_paths:
        print(f"[TEST] Scenario A: Simple function test")
        print(f"[TEST] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[FAIL] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"[TEST] Function: {outer_data.get('func_name', 'load_and_preprocess_data')}")
        print(f"[TEST] Args: {len(outer_args)} positional arguments")
        print(f"[TEST] Kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function
        try:
            print("[TEST] Executing load_and_preprocess_data...")
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("[TEST] Function executed successfully")
        except Exception as e:
            print(f"[FAIL] Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            print("[TEST] Comparing results with expected output...")
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("[TEST] TEST PASSED")
                sys.exit(0)
            else:
                print(f"[FAIL] Result mismatch: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (both outer and inner data exist)
    elif outer_path and inner_paths:
        print(f"[TEST] Scenario B: Factory/Closure pattern test")
        print(f"[TEST] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[FAIL] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract outer args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[TEST] Creating operator with outer args...")
        
        # Phase 1: Reconstruct the operator
        try:
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("[TEST] Operator created successfully")
        except Exception as e:
            print(f"[FAIL] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[FAIL] Operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[TEST] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[FAIL] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print(f"[TEST] Executing operator with inner args...")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("[TEST] Operator executed successfully")
            except Exception as e:
                print(f"[FAIL] Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                print("[TEST] Comparing results with expected output...")
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print("[TEST] TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"[FAIL] Result mismatch: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"[FAIL] Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        print(f"[FAIL] Could not determine test scenario from data paths: {data_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()