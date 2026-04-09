import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/IHCP_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path is None:
        print("ERROR: Could not find outer data file.")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if we have inner paths (Factory/Closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected Factory/Closure pattern with {len(inner_paths)} inner data file(s).")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from outer function, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator
                print("Executing agent_operator...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Agent operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR in inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    # Comparison phase
    try:
        print("Comparing results...")
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()