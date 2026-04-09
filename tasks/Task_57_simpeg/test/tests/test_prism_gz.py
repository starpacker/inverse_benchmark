import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_prism_gz import prism_gz
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_prism_gz.pkl']

def main():
    """Main test function for prism_gz."""
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_prism_gz.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_prism_gz.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute prism_gz with outer args
    try:
        agent_result = prism_gz(*outer_args, **outer_kwargs)
        print(f"Executed prism_gz successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute prism_gz: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from prism_gz, got {type(agent_result)}")
            sys.exit(1)
        
        print(f"Scenario B detected: Testing closure/operator pattern")
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Executed operator successfully")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print(f"Scenario A detected: Simple function test")
        
        expected = outer_data.get('output')
        result = agent_result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()