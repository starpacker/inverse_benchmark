import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_gradient_data_fidelity import gradient_data_fidelity

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for gradient_data_fidelity."""
    
    # Data paths provided
    data_paths = ['/data/yjh/direct_mri_sandbox_sandbox/run_code/std_data/standard_data_gradient_data_fidelity.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_gradient_data_fidelity.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_gradient_data_fidelity.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Determine scenario based on presence of inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Phase 1: Reconstruct Operator
        try:
            agent_operator = gradient_data_fidelity(*outer_args, **outer_kwargs)
            print(f"Successfully created agent_operator: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify it's callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"\nProcessing inner path: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed agent_operator, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner path: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("\n=== Scenario A: Simple Function ===")
        
        # Execute the function
        try:
            result = gradient_data_fidelity(*outer_args, **outer_kwargs)
            print(f"Successfully executed gradient_data_fidelity, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute gradient_data_fidelity: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("Verification passed")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()