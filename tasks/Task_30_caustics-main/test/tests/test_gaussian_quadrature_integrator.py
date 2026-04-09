import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_gaussian_quadrature_integrator import gaussian_quadrature_integrator
from verification_utils import recursive_check

def test_gaussian_quadrature_integrator():
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_gaussian_quadrature_integrator.pkl']
    
    # Identify data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if 'standard_data_gaussian_quadrature_integrator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_gaussian_quadrature_integrator_' in path:
            inner_path = path

    if not outer_path:
        print("Error: Standard data file not found for gaussian_quadrature_integrator.")
        sys.exit(1)

    try:
        # Load outer data (Arguments for the main function)
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print(f"Running gaussian_quadrature_integrator with loaded data...")
        
        # Scenario: Simple execution (Scenario A from instructions)
        # Based on the function signature provided: def gaussian_quadrature_integrator(F: Tensor, weight: Tensor) -> Tensor
        # It calculates a value immediately, it does not return a closure/operator.
        
        actual_result = gaussian_quadrature_integrator(*outer_args, **outer_kwargs)

        # Verification
        passed, msg = recursive_check(expected_output, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_gaussian_quadrature_integrator()