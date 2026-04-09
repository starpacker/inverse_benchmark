import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_make_sources import make_sources
from verification_utils import recursive_check

def main():
    """Main test function for make_sources."""
    
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_make_sources.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_sources.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_make_sources.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute the function to get the operator/result
    try:
        agent_result = make_sources(*outer_args, **outer_kwargs)
        print("Successfully executed make_sources with outer args/kwargs")
    except Exception as e:
        print(f"ERROR: Failed to execute make_sources: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if we have inner data (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable from make_sources (factory pattern), but got non-callable")
            sys.exit(1)
        
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
                print("Successfully executed the operator with inner args/kwargs")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function
        # The result from Phase 1 is the actual result
        expected = outer_data.get('output')
        result = agent_result
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("TEST PASSED")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()