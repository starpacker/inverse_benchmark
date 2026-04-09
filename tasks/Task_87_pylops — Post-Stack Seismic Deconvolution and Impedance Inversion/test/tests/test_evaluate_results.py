import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pylops_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Determine outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute the function
    try:
        print("Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Handle inner paths if present (factory/closure pattern)
    if inner_paths:
        print(f"Found {len(inner_paths)} inner data file(s) - Factory/Closure pattern detected")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable but inner paths exist. Treating as simple function.")
        else:
            # Load and execute with inner data
            for inner_path in inner_paths:
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output')
                    
                    print("Executing operator with inner data...")
                    actual_result = result(*inner_args, **inner_kwargs)
                    
                    # Compare with inner output
                    passed, msg = recursive_check(inner_output, actual_result)
                    
                    if not passed:
                        print(f"TEST FAILED (inner execution): {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner execution verification passed.")
                        
                except Exception as e:
                    print(f"ERROR processing inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    # Scenario A: Simple function - compare result with outer output
    print("Simple function scenario - comparing result with expected output")
    
    try:
        passed, msg = recursive_check(outer_output, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()