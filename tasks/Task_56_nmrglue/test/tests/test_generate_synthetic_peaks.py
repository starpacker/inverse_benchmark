import sys
import os
import dill
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/nmrglue_sandbox_sandbox/run_code')

import numpy as np

# Import the target function
from agent_generate_synthetic_peaks import generate_synthetic_peaks

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for generate_synthetic_peaks."""
    
    # Data paths provided
    data_paths = ['/data/yjh/nmrglue_sandbox_sandbox/run_code/std_data/standard_data_generate_synthetic_peaks.pkl']
    
    # Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner (parent_function) file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer file (exact match for function name)
        elif basename == 'standard_data_generate_synthetic_peaks.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_synthetic_peaks.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Loaded outer data for function: {outer_data.get('func_name')}")
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function with outer args/kwargs
    try:
        agent_result = generate_synthetic_peaks(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        print(f"Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute generate_synthetic_peaks: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        # Verify the result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Load inner data and execute
        try:
            inner_path = inner_paths[0]  # Use first inner path
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Loaded inner data for function: {inner_data.get('func_name')}")
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner args/kwargs
            result = agent_operator(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"ERROR: Failed to execute inner operation: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        result = agent_result
        expected = outer_output
    
    # Phase 3: Comparison
    print("\n--- Verification ---")
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(result)}")
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print(f"\nTEST FAILED: {msg}")
            
            # Additional debug info
            if isinstance(expected, list) and isinstance(result, list):
                print(f"Expected length: {len(expected)}, Result length: {len(result)}")
                if len(expected) > 0 and len(result) > 0:
                    print(f"First expected element: {expected[0]}")
                    print(f"First result element: {result[0]}")
            
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()