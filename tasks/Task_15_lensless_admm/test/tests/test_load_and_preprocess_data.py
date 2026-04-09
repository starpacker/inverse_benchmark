import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_admm_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and run the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
        # Phase 2: Check if this is a factory pattern or simple function
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
            
            # Verify the result is callable
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator from factory, got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                
                if not os.path.exists(inner_path):
                    print(f"ERROR: Inner data file does not exist: {inner_path}")
                    sys.exit(1)
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args: {len(inner_args)} positional arguments")
                print(f"Inner kwargs: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing operator with inner args/kwargs...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {inner_path}")
        
        else:
            # Scenario A: Simple function
            print("Detected simple function pattern (no inner data files)")
            
            result = agent_result
            expected = outer_output
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Exception during test execution")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()