import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def load_data(file_path):
    """Load pickled data from file."""
    with open(file_path, 'rb') as f:
        return dill.load(f)


def main():
    # Data paths provided
    data_paths = ['/home/yjh/ad_pty/code_2/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_evaluate_results.pkl':
                outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("\n=== Phase 1: Loading outer data ===")
        outer_data = load_data(outer_path)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\n=== Executing evaluate_results ===")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Verify the result is callable
        if not callable(result):
            print(f"WARNING: Result is not callable, treating as Scenario A")
            # Fall through to Scenario A logic
        else:
            agent_operator = result
            
            for inner_path in inner_paths:
                try:
                    print(f"\n--- Processing inner data: {inner_path} ---")
                    inner_data = load_data(inner_path)
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output')
                    
                    print(f"Inner args count: {len(inner_args)}")
                    print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                    # Execute the operator
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Compare results
                    passed, msg = recursive_check(expected, actual_result)
                    
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner test passed: {msg}")
                        
                except Exception as e:
                    print(f"ERROR: Failed processing inner data {inner_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("\nTEST PASSED")
            sys.exit(0)
    
    # Scenario A: Simple function (or fallback from Scenario B)
    print("\n=== Scenario A: Simple Function ===")
    
    try:
        expected = outer_output
        
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        # Compare results
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"Comparison result: {msg}")
            print("\nTEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR: Failed during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()