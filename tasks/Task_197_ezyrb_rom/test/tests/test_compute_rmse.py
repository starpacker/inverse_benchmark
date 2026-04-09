import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_rmse import compute_rmse

# Import verification utility
from verification_utils import recursive_check


def main():
    """
    Unit test for compute_rmse function.
    Handles both Scenario A (simple function) and Scenario B (factory/closure pattern).
    """
    
    # Data paths provided
    data_paths = ['/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_compute_rmse.pkl']
    
    # Step 1: Analyze data files to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print("ERROR: Data file not found: " + path)
            sys.exit(1)
        
        # Check if this is the outer data file (exact match)
        if path.endswith('standard_data_compute_rmse.pkl') and 'parent_function' not in path:
            outer_path = path
        # Check if this is inner data (contains parent_function)
        elif 'parent_function' in path and 'compute_rmse' in path:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_rmse.pkl)")
        sys.exit(1)
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    try:
        # Phase 1: Load outer data and reconstruct operator/result
        print("Loading outer data from: " + outer_path)
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print("Executing compute_rmse with outer data...")
        agent_result = compute_rmse(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine final result and expected output
        if is_factory_pattern:
            # Scenario B: Factory/Closure Pattern
            print("Detected Scenario B: Factory/Closure Pattern")
            
            if not callable(agent_result):
                print("ERROR: Expected callable operator from compute_rmse, got: " + str(type(agent_result)))
                sys.exit(1)
            
            # Load inner data
            inner_path = inner_paths[0]
            print("Loading inner data from: " + inner_path)
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print("Executing agent operator with inner data...")
            final_result = agent_result(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function
            print("Detected Scenario A: Simple Function")
            final_result = agent_result
            expected_output = outer_output
        
        # Phase 3: Verification
        print("Verifying results...")
        passed, message = recursive_check(expected_output, final_result)
        
        if not passed:
            print("TEST FAILED")
            print("Verification message: " + str(message))
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print("TEST FAILED WITH EXCEPTION")
        print("Error: " + str(e))
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()