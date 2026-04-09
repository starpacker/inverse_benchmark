import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_shepp_logan_phantom import shepp_logan_phantom

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for shepp_logan_phantom."""
    
    # Data paths provided
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_shepp_logan_phantom.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_shepp_logan_phantom.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_shepp_logan_phantom.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
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
        print("Executing shepp_logan_phantom with outer args/kwargs...")
        agent_operator = shepp_logan_phantom(*outer_args, **outer_kwargs)
        print(f"Result type: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute shepp_logan_phantom: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing agent_operator with inner args/kwargs...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for inner file {os.path.basename(inner_path)}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner data processing: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function test")
        
        result = agent_operator
        expected = outer_output
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: Failed during result comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()