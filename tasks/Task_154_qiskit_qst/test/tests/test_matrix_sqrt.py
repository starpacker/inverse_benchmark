import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_matrix_sqrt import matrix_sqrt
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_matrix_sqrt.pkl']

def main():
    try:
        # Separate outer and inner data paths
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_matrix_sqrt.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_matrix_sqrt.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Executing matrix_sqrt with outer args...")
        agent_result = matrix_sqrt(*outer_args, **outer_kwargs)
        
        # Phase 2: Check if we have inner data (factory pattern) or simple function
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
            
            if not callable(agent_result):
                print(f"ERROR: Expected callable from matrix_sqrt, got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Executing agent_operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
        else:
            # Scenario A: Simple Function
            print("Detected simple function pattern (no inner data)")
            result = agent_result
            expected = outer_data.get('output')
            
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()