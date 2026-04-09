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
    data_paths = ['/data/yjh/direct_mri_sandbox_sandbox/run_code/std_data/standard_data_shepp_logan_phantom.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        
        # Check if it's an inner (parent_function) file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the outer file (exact match for the function)
        elif basename == 'standard_data_shepp_logan_phantom.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_shepp_logan_phantom.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Successfully loaded outer data file")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the target function
    try:
        agent_result = shepp_logan_phantom(*outer_args, **outer_kwargs)
        print(f"Successfully executed shepp_logan_phantom")
        print(f"Result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute shepp_logan_phantom: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on presence of inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("\nScenario B: Factory/Closure Pattern detected")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable result for closure pattern, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Successfully loaded inner data file")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner arguments
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Successfully executed operator with inner args")
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
                else:
                    print(f"Inner test passed: {msg}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\nScenario A: Simple Function detected")
        
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Verification: {msg}")
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()