import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/sigpy_mri_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Identify outer and inner paths
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
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Run outer function to get the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Outer function executed successfully, result type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to execute outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data for function: {inner_data.get('func_name', 'unknown')}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner arguments
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner function executed successfully, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        
        # Execute the function
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Function executed successfully, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()