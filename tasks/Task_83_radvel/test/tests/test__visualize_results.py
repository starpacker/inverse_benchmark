import sys
import os
import dill
import traceback

# Add the path to find the module
sys.path.insert(0, '/data/yjh/radvel_sandbox_sandbox/run_code')

# Import the target function
from agent__visualize_results import _visualize_results
from verification_utils import recursive_check

def main():
    """Main test function for _visualize_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/radvel_sandbox_sandbox/run_code/std_data/standard_data__visualize_results.pkl']
    
    # Analyze paths to determine test strategy
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data__visualize_results.pkl':
            outer_path = path
    
    print(f"[INFO] Outer path: {outer_path}")
    print(f"[INFO] Inner path: {inner_path}")
    
    # Scenario A: Simple Function (no inner path)
    if outer_path and not inner_path:
        print("[INFO] Detected Scenario A: Simple Function")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"[INFO] Loaded outer data from {outer_path}")
        except Exception as e:
            print(f"[FAIL] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"[INFO] Function name: {outer_data.get('func_name')}")
        print(f"[INFO] Number of args: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        try:
            # Execute the function
            result = _visualize_results(*args, **kwargs)
            print(f"[INFO] Function executed successfully")
        except Exception as e:
            print(f"[FAIL] Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("[INFO] TEST PASSED")
                sys.exit(0)
            else:
                print(f"[FAIL] Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure Pattern
    elif outer_path and inner_path:
        print("[INFO] Detected Scenario B: Factory/Closure Pattern")
        
        # Phase 1: Load outer data and create operator
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print(f"[INFO] Loaded outer data from {outer_path}")
        except Exception as e:
            print(f"[FAIL] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        try:
            agent_operator = _visualize_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Created agent operator")
        except Exception as e:
            print(f"[FAIL] Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[FAIL] Agent operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"[INFO] Loaded inner data from {inner_path}")
        except Exception as e:
            print(f"[FAIL] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"[INFO] Agent operator executed successfully")
        except Exception as e:
            print(f"[FAIL] Agent operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("[INFO] TEST PASSED")
                sys.exit(0)
            else:
                print(f"[FAIL] Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print("[FAIL] Could not determine test scenario from data paths")
        sys.exit(1)


if __name__ == '__main__':
    main()