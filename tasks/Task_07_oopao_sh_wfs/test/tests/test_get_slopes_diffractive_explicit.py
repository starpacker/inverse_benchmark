import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_get_slopes_diffractive_explicit import get_slopes_diffractive_explicit
from verification_utils import recursive_check

def main():
    """Main test function for get_slopes_diffractive_explicit."""
    
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data_get_slopes_diffractive_explicit.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_get_slopes_diffractive_explicit.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_get_slopes_diffractive_explicit.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing get_slopes_diffractive_explicit with outer args/kwargs...")
        agent_result = get_slopes_diffractive_explicit(*outer_args, **outer_kwargs)
        print(f"Agent result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR executing get_slopes_diffractive_explicit: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Check if agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable agent_result for factory pattern, got {type(agent_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing agent_result (operator) with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for inner path {inner_path}")
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
        # Scenario A: Simple Function
        print("Detected simple function pattern (no inner data files)")
        
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            print("Comparing results...")
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