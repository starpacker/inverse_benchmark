import sys
import os
import dill
import numpy as np
import scipy.sparse as sp
import traceback

# Ensure the import path is correct for the target function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_assemble_fem import assemble_fem
from verification_utils import recursive_check

def convert_if_sparse(obj):
    """
    Converts scipy sparse matrices to a dictionary structure containing their 
    underlying numpy arrays. This allows recursive_check (which typically 
    handles dicts and numpy arrays) to verify the contents without tripping 
    over sparse matrix boolean ambiguity.
    """
    if sp.issparse(obj):
        # Ensure we are comparing canonical representations if possible, 
        # though assemble_fem returns csc explicitly.
        return {
            '__is_sparse__': True,
            'format': obj.getformat(),
            'data': obj.data,
            'indices': obj.indices,
            'indptr': obj.indptr,
            'shape': obj.shape
        }
    return obj

def test_assemble_fem():
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_assemble_fem.pkl']
    
    outer_path = None
    inner_path = None

    # Identify file roles
    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        else:
            outer_path = path

    if not outer_path:
        print("Error: No standard_data_assemble_fem.pkl found.")
        sys.exit(1)

    try:
        print(f"Loading Outer Data: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        print("Executing assemble_fem...")
        # assemble_fem is a direct function, not a factory in the provided code,
        # but we handle the structure just in case the context implies otherwise.
        # Based on the prompt, it returns the matrix directly.
        actual_result = assemble_fem(*outer_args, **outer_kwargs)

        if inner_path:
            # Scenario B: assemble_fem returned a callable (not the case here based on code, but keeping logic)
            print(f"Loading Inner Data: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            if not callable(actual_result):
                 print("Error: Inner data provided but result is not callable.")
                 sys.exit(1)
                 
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            actual_result = actual_result(*inner_args, **inner_kwargs)
            expected_result = inner_data['output']
        else:
            # Scenario A: Direct comparison
            print("No inner data found. Comparing direct output of assemble_fem.")
            expected_result = outer_data['output']

        print("Verifying results...")
        
        # FIX: Convert sparse matrices to dictionaries of arrays before checking
        # to avoid "The truth value of an array with more than one element is ambiguous"
        safe_expected = convert_if_sparse(expected_result)
        safe_actual = convert_if_sparse(actual_result)

        passed, msg = recursive_check(safe_expected, safe_actual)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print("An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_assemble_fem()