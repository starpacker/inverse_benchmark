import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/suncasa_radio_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing load_and_preprocess_data with outer arguments...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Check if agent_result is callable
        if not callable(agent_result):
            print("WARNING: agent_result is not callable. Treating as Scenario A.")
            result = agent_result
            expected = outer_output
        else:
            # Load inner data and execute the operator
            inner_path = inner_paths[0]  # Use first inner path
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner arguments
            try:
                print("Executing operator with inner arguments...")
                result = agent_result(*inner_args, **inner_kwargs)
                expected = inner_output
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        result = agent_result
        expected = outer_output
    
    # Phase 3: Verification
    try:
        print("Verifying results...")
        passed, msg = recursive_check(expected, result)
        
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

if __name__ == "__main__":
    main()