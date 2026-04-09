import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_interaction_param import interaction_param
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/abtem_sim_sandbox_sandbox/run_code/std_data/standard_data_interaction_param.pkl']

def main():
    try:
        # Separate outer and inner paths
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            else:
                outer_path = path
        
        # Phase 1: Load outer data and reconstruct operator
        if outer_path is None:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the target function
        print("Executing interaction_param with outer args/kwargs...")
        agent_result = interaction_param(*outer_args, **outer_kwargs)
        print(f"Agent result type: {type(agent_result)}")
        
        # Phase 2: Check if we have inner paths (factory/closure pattern)
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print(f"Detected factory/closure pattern with {len(inner_paths)} inner data file(s)")
            
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator but got {type(agent_result)}")
                sys.exit(1)
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator with inner args
                print("Executing agent_operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for inner path: {inner_path}")
                    print(f"Failure message: {msg}")
                    print(f"Expected type: {type(expected)}, Result type: {type(result)}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
        else:
            # Scenario A: Simple function
            print("Detected simple function pattern (no inner data)")
            result = agent_result
            expected = outer_output
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                print(f"Expected: {expected}")
                print(f"Result: {result}")
                print(f"Expected type: {type(expected)}, Result type: {type(result)}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()