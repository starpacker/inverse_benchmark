import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pfsspy_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Scenario A: Simple function (only outer data exists)
    if outer_path and not inner_paths:
        print(f"[INFO] Scenario A detected: Simple function test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args, kwargs, and expected output
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        expected = outer_data.get('output')
        
        print(f"[INFO] Running forward_operator with {len(args)} args and {len(kwargs)} kwargs")
        
        try:
            result = forward_operator(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify result
        print("[INFO] Comparing result with expected output...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"[ERROR] Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"[FAIL] Verification failed: {msg}")
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (both outer and inner data exist)
    elif outer_path and inner_paths:
        print(f"[INFO] Scenario B detected: Factory/Closure pattern test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 1: Reconstruct operator
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[INFO] Creating operator with forward_operator(*args, **kwargs)")
        
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[ERROR] Created operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"[INFO] Executing operator with inner args/kwargs")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            print("[INFO] Comparing result with expected output...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"[ERROR] Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"[FAIL] Verification failed for {inner_path}: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print("[ERROR] No valid data files found")
        sys.exit(1)

if __name__ == "__main__":
    main()