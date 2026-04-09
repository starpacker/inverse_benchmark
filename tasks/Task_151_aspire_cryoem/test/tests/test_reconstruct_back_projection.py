import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_reconstruct_back_projection import reconstruct_back_projection
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_reconstruct_back_projection.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_reconstruct_back_projection.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_reconstruct_back_projection.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing reconstruct_back_projection with outer args/kwargs...")
        agent_result = reconstruct_back_projection(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR executing reconstruct_back_projection: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if the result is callable (operator/closure)
        if not callable(agent_result):
            print("WARNING: Result is not callable but inner paths exist. Treating as Scenario A.")
            # Fall through to Scenario A logic
            result = agent_result
            expected = outer_output
        else:
            # Load inner data and execute the operator
            inner_path = inner_paths[0]  # Use the first inner path
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing agent_operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully. Result type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function, no inner data files")
        result = agent_result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Running verification...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()