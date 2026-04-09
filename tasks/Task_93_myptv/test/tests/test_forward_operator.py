import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def fix_seeds(seed=42):
    """Fix random seeds for reproducibility"""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass

def main():
    data_paths = ['/data/yjh/myptv_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        # Check for inner data (factory/closure pattern)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check for outer data (exact match for forward_operator)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Fix seeds before calling to match the original recording
    fix_seeds(42)
    
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory/closure pattern (Scenario B) with {len(inner_paths)} inner data file(s)")
        
        # The result from Phase 1 should be callable (the operator)
        if not callable(result):
            print("ERROR: Expected forward_operator to return a callable for factory pattern")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            # Fix seeds again before inner call
            fix_seeds(42)
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare inner result with expected
            passed, msg = recursive_check(inner_expected, inner_result)
            if not passed:
                print("TEST FAILED")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed for: {inner_path}")
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern (Scenario A)")
        
        # Compare result with expected output
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print("TEST FAILED")
            print(f"Mismatch details: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()