import sys
import os
import dill
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    data_paths = ['/data/yjh/pyxu_imaging_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run forward_operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded. Function name: {outer_data.get('func_name')}")
        print(f"Number of args: {len(outer_args)}, Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute forward_operator with outer args/kwargs
    try:
        print("Executing forward_operator with outer args/kwargs...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario: Check if there are inner paths (factory pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded. Function name: {inner_data.get('func_name')}")
                print(f"Number of inner args: {len(inner_args)}, Number of inner kwargs: {len(inner_kwargs)}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args/kwargs
            try:
                print("Executing agent_operator with inner args/kwargs...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully. Result type: {type(actual_result)}")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                print("\nComparing results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\nScenario A detected: Simple function call")
        
        expected = outer_output
        
        # Compare results
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()