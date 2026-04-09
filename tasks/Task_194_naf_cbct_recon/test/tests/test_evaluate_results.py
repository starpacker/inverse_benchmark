import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Test the evaluate_results function."""
    
    # Data paths provided
    data_paths = ['/data/yjh/naf_cbct_recon_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\n=== Phase 2: Executing evaluate_results ===")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine which scenario we're in
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        # Check if result is callable (an operator)
        if callable(result):
            print("Result is callable (operator/closure)")
            agent_operator = result
            
            # Process inner paths
            for inner_path in inner_paths:
                try:
                    print(f"\nProcessing inner data: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output', None)
                    
                    print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                    
                    # Execute the operator with inner args
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Verify results
                    passed, msg = recursive_check(expected, actual_result)
                    
                    if not passed:
                        print(f"VERIFICATION FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner execution verified successfully")
                        
                except Exception as e:
                    print(f"ERROR processing inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
        else:
            # Result is not callable, but we have inner paths - unexpected
            print("WARNING: Inner paths exist but result is not callable")
            print("Falling back to direct comparison with outer output")
            expected = outer_output
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n=== Scenario A: Simple Function ===")
        expected = outer_output
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print("Direct comparison verified successfully")
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*50)
    print("TEST PASSED")
    print("="*50)
    sys.exit(0)

if __name__ == "__main__":
    main()