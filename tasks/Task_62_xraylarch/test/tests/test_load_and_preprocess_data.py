import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/xraylarch_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Classify data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Scenario A: Simple function test
    if outer_path and not inner_paths:
        print(f"[TEST] Scenario A: Simple function test")
        print(f"[TEST] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"[TEST] Outer args: {len(outer_args)} positional arguments")
        print(f"[TEST] Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function
        try:
            print("[TEST] Executing load_and_preprocess_data...")
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("[TEST] Function executed successfully")
        except Exception as e:
            print(f"[ERROR] Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            print("[TEST] Comparing results...")
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("[TEST] TEST PASSED")
                sys.exit(0)
            else:
                print(f"[TEST] TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern
    elif outer_path and inner_paths:
        print(f"[TEST] Scenario B: Factory/Closure pattern")
        print(f"[TEST] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Phase 1: Create the operator
        try:
            print("[TEST] Creating operator from load_and_preprocess_data...")
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("[TEST] Operator created successfully")
        except Exception as e:
            print(f"[ERROR] Operator creation failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[ERROR] Created operator is not callable: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[TEST] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                print("[TEST] Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("[TEST] Operator executed successfully")
            except Exception as e:
                print(f"[ERROR] Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                print("[TEST] Comparing results...")
                passed, msg = recursive_check(expected_output, result)
                if passed:
                    print("[TEST] TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"[TEST] TEST FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        print("[ERROR] No valid data files found")
        sys.exit(1)

if __name__ == "__main__":
    main()