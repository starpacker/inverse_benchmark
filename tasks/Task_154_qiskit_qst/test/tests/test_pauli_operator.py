import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_pauli_operator import pauli_operator
from verification_utils import recursive_check

def main():
    """Main test function for pauli_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_pauli_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_pauli_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_pauli_operator.pkl)")
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
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute pauli_operator with outer args
    try:
        agent_result = pauli_operator(*outer_args, **outer_kwargs)
        print(f"Successfully executed pauli_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute pauli_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Verify that agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from pauli_operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner args
            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"Successfully executed agent_operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for inner data {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
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
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                print(f"Expected type: {type(expected)}")
                print(f"Result type: {type(result)}")
                if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
                    print(f"Expected shape: {expected.shape}, Result shape: {result.shape}")
                    print(f"Expected dtype: {expected.dtype}, Result dtype: {result.dtype}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()