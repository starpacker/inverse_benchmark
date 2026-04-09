import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def fix_seeds(seed=42):
    """Fix random seeds for reproducibility."""
    import random
    np.random.seed(seed)
    random.seed(seed)

def main():
    """Main test function."""
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
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
    expected_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Fix seeds before running to ensure reproducibility
    fix_seeds(42)
    
    # Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and we have inner paths)
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure pattern
        print("Detected factory pattern with inner paths")
        
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
            expected_output = inner_data.get('output', None)
            
            # Execute the operator/closure
            try:
                result = result(*inner_args, **inner_kwargs)
                print("Inner function executed successfully")
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function, result is already computed
        print("Detected simple function pattern")
    
    # Phase 2: Verification
    if expected_output is None:
        print("WARNING: No expected output found in data file")
        print("TEST PASSED (no expected output to compare)")
        sys.exit(0)
    
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()