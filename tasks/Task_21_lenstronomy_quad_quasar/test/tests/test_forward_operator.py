import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    data_paths = ['/home/yjh/lenstronomy_quad_quasar_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_forward_operator.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute forward_operator
    try:
        print("Executing forward_operator...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully, result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # The result from outer call should be callable
        if not callable(result):
            print(f"ERROR: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing agent_operator with inner data...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing inner results...")
                passed, msg = recursive_check(inner_expected, inner_result)
                
                if not passed:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function - compare result directly with expected output
        print("\nScenario A detected: Simple function call")
        
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
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