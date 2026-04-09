import sys
import os
import dill
import traceback
import numpy as np

# Add the working directory to path
sys.path.insert(0, '/data/yjh/pyidi_dic_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"WARNING: Path does not exist: {path}")
            continue
            
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute the function
    try:
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\n=== Executing evaluate_results ===")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Handle based on whether we have inner paths (factory pattern) or not
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Phase 2: Factory/Closure Pattern Detected ===")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from outer function, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"Number of inner args: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                print("\nExecuting agent_operator with inner args...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("\n=== Comparing Results ===")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
                    
            except Exception as e:
                print(f"ERROR processing inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\n=== Phase 2: Simple Function Pattern ===")
        
        expected = outer_output
        
        # Compare results
        print("\n=== Comparing Results ===")
        print(f"Expected type: {type(expected)}")
        print(f"Actual type: {type(result)}")
        
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                # Print more details about the mismatch
                if isinstance(expected, dict) and isinstance(result, dict):
                    print("\nExpected keys:", list(expected.keys()))
                    print("Actual keys:", list(result.keys()))
                    for key in expected:
                        if key in result:
                            if expected[key] != result[key]:
                                print(f"  Key '{key}': expected={expected[key]}, actual={result[key]}")
                        else:
                            print(f"  Key '{key}' missing in result")
                sys.exit(1)
            else:
                print(f"TEST PASSED: {msg}")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()