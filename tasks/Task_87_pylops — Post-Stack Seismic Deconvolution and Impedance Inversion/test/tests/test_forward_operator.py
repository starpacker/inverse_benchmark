import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pylops_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if not inner_paths:
        print("Scenario A: Simple Function Test")
        
        if outer_path is None:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        try:
            # Load outer data
            print(f"Loading data from: {outer_path}")
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Function name: {outer_data.get('func_name', 'unknown')}")
            print(f"Number of args: {len(args)}")
            print(f"Number of kwargs: {len(kwargs)}")
            
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            print("Executing forward_operator...")
            result = forward_operator(*args, **kwargs)
            print("Execution completed.")
            
        except Exception as e:
            print(f"ERROR: Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Verify results
            print("Verifying results...")
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure Pattern
    else:
        print("Scenario B: Factory/Closure Pattern Test")
        
        if outer_path is None:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        # Phase 1: Reconstruct Operator
        try:
            print(f"Phase 1: Loading outer data from: {outer_path}")
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Creating operator with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print("ERROR: forward_operator did not return a callable.")
                sys.exit(1)
            
            print("Operator created successfully.")
            
        except Exception as e:
            print(f"ERROR: Failed in Phase 1: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data and verify
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\nPhase 2: Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Executing operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Execution completed.")
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print(f"Test for {os.path.basename(inner_path)}: PASSED")
                else:
                    print(f"Test for {os.path.basename(inner_path)}: FAILED - {msg}")
                    all_passed = False
                    
            except Exception as e:
                print(f"ERROR: Failed processing {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()