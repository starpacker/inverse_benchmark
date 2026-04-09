import sys
import os
import dill
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Determine test scenario by analyzing file paths
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
    
    # Phase 1: Load outer data and run the function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Execute the forward_operator function
    print("Executing forward_operator with outer data...")
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if there are inner paths (factory pattern)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output', None)
            
            print(f"Inner data function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Number of inner args: {len(inner_args)}")
            print(f"Number of inner kwargs: {len(inner_kwargs)}")
            
            # Execute the agent operator with inner args
            print("Executing agent_operator with inner data...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_output, actual_result)
            except Exception as e:
                print(f"ERROR: Comparison failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - compare result directly with outer output
        print("Detected simple function pattern (no inner data)")
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(outer_output, result)
        except Exception as e:
            print(f"ERROR: Comparison failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)

if __name__ == "__main__":
    main()