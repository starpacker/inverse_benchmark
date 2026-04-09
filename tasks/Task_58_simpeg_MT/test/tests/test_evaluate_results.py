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
    data_paths = ['/data/yjh/simpeg_MT_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    print(f"[TEST] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[TEST] Outer args count: {len(outer_args)}")
    print(f"[TEST] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("[TEST] Detected Scenario B: Factory/Closure Pattern")
        
        try:
            print("[TEST] Running evaluate_results to get operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print(f"ERROR: Result is not callable. Type: {type(agent_operator)}")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]
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
        expected = inner_data.get('output')
        
        print(f"[TEST] Inner args count: {len(inner_args)}")
        print(f"[TEST] Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        try:
            print("[TEST] Executing operator with inner args...")
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("[TEST] Detected Scenario A: Simple Function")
        
        try:
            print("[TEST] Running evaluate_results...")
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = expected_output
    
    # Phase 2: Verification
    print("[TEST] Verifying results...")
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"[DEBUG] Expected type: {type(expected)}")
        print(f"[DEBUG] Result type: {type(result)}")
        if isinstance(expected, dict) and isinstance(result, dict):
            print(f"[DEBUG] Expected keys: {list(expected.keys())}")
            print(f"[DEBUG] Result keys: {list(result.keys())}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()