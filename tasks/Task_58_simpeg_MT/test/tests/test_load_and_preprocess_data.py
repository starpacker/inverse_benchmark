import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/simpeg_MT_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        # Verify that the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Executing inner operator...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for inner data {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        
        result = agent_result
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()