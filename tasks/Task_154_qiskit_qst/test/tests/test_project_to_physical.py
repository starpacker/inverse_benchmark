import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_project_to_physical import project_to_physical

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for project_to_physical."""
    
    # Define data paths
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_project_to_physical.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_project_to_physical.pkl':
            outer_path = path
    
    # Verify we have the outer data file
    if outer_path is None:
        print("ERROR: Could not find standard_data_project_to_physical.pkl")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Load outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Phase 1: Execute the function with outer data
        print("\nPhase 1: Executing project_to_physical with outer arguments...")
        result = project_to_physical(*outer_args, **outer_kwargs)
        print("  Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute project_to_physical: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine test scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        # Check if result is callable (an operator)
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
            
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"  Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Inner args count: {len(inner_args)}")
                print(f"  Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Phase 2: Execute the operator with inner arguments
                print("\nPhase 2: Executing operator with inner arguments...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("  Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                print("\nVerifying results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for inner data: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        
        expected = outer_output
        actual_result = result
        
        # Verify results
        try:
            print("\nVerifying results...")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("  Verification passed.")
                
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()