import sys
import os
import dill
import traceback
import numpy as np

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Step 1: Analyze data files to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer path (exact match for forward_operator)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args types: {[type(a).__name__ for a in outer_args]}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Step 3: Execute forward_operator with outer data
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"Successfully executed forward_operator")
        print(f"Result type: {type(result).__name__}")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine test scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Check if result is callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable result for closure pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nProcessing inner path: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args types: {[type(a).__name__ for a in inner_args]}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed agent_operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed: {msg}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\n=== Scenario A: Simple Function ===")
        
        expected = outer_output
        actual_result = result
        
        print(f"Expected type: {type(expected).__name__}")
        print(f"Actual result type: {type(actual_result).__name__}")
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Verification message: {msg}")
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()