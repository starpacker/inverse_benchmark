import sys
import os
import dill
import torch
import numpy as np
import traceback
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Ensure the module containing the function is in the python path
# Assuming the file structure allows direct import or adjacent file access
try:
    from agent_forward_operator import forward_operator
except ImportError:
    # If strictly local, we might need to adjust path, but assuming environment is set
    pass

from verification_utils import recursive_check

def test_forward_operator():
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 1. Identify File Roles
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'standard_data_forward_operator.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_forward_operator_' in p:
            inner_paths.append(p)
            
    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found in paths.")
        sys.exit(1)
        
    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)
        
    # 2. Execution Strategy
    # Scenario A: The function returns data directly (no inner paths found)
    # Scenario B: The function returns a closure/factory (inner paths exist)
    
    try:
        # Phase 1: Execute the primary function
        print("Executing forward_operator with loaded arguments...")
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Determine if we are in Scenario A or B based on available data files
        if not inner_paths:
            # Scenario A: Direct Execution
            actual_result = forward_operator(*args, **kwargs)
            expected_result = outer_data['output']
            
            print("Verifying result (Scenario A - Direct Output)...")
            passed, msg = recursive_check(expected_result, actual_result)
            
            if not passed:
                print(f"Verification Failed:\n{msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        else:
            # Scenario B: Factory Pattern
            # Note: Based on the provided code in the prompt, forward_operator returns (J, data_amp) directly.
            # However, if the data capture logic detected inner calls, we would process them here.
            # Given the provided path list only has the outer file, this block is technically unreachable 
            # with the current specific input, but written for robustness as requested.
            
            agent_operator = forward_operator(*args, **kwargs)
            
            # Verify the operator was created successfully
            if not callable(agent_operator):
                print("Error: Expected a callable (closure) from forward_operator, but got:", type(agent_operator))
                sys.exit(1)
                
            print(f"Operator created. Testing {len(inner_paths)} inner execution scenarios...")
            
            for i_path in inner_paths:
                print(f"  Loading Inner Data: {i_path}")
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                i_expected = inner_data['output']
                
                i_actual = agent_operator(*i_args, **i_kwargs)
                
                passed, msg = recursive_check(i_expected, i_actual)
                if not passed:
                    print(f"  Inner Verification Failed for {i_path}:\n{msg}")
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)

    except Exception as e:
        print("An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()