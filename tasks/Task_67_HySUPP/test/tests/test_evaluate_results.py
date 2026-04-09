import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        print(f"[INFO] Scenario A: Simple function test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
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
        
        print(f"[INFO] Running evaluate_results with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        print("[INFO] Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern
    elif outer_path and inner_paths:
        print(f"[INFO] Scenario B: Factory/Closure pattern test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs for creating the operator
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[INFO] Creating operator with evaluate_results...")
        
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"[ERROR] Created operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner path
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
            expected_output = inner_data.get('output')
            
            print(f"[INFO] Executing operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("[INFO] Comparing results...")
            try:
                passed, msg = recursive_check(expected_output, result)
            except Exception as e:
                print(f"[ERROR] Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print("[ERROR] No valid data paths found")
        sys.exit(1)

if __name__ == "__main__":
    main()