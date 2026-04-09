import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_calculate_so2 import calculate_so2
from verification_utils import recursive_check

def main():
    """Main test function for calculate_so2."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_calculate_so2.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_calculate_so2.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            # Extract args and kwargs
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Args type: {type(outer_args)}, length: {len(outer_args) if isinstance(outer_args, (list, tuple)) else 'N/A'}")
            
        except Exception as e:
            print(f"Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            result = calculate_so2(*outer_args, **outer_kwargs)
            print(f"Function executed successfully")
            print(f"Result type: {type(result)}")
            
        except Exception as e:
            print(f"Failed to execute calculate_so2: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (inner paths exist)
    elif outer_path and inner_paths:
        try:
            # Phase 1: Load outer data and create operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            
            # Create the operator/closure
            agent_operator = calculate_so2(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: calculate_so2 did not return a callable, got {type(agent_operator)}")
                sys.exit(1)
            
            print("Agent operator created successfully")
            
        except Exception as e:
            print(f"Failed in Phase 1 (operator creation): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Loaded inner data from: {inner_path}")
                
                # Execute the operator
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Failed processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print("ERROR: No valid data paths found")
        sys.exit(1)

if __name__ == "__main__":
    main()