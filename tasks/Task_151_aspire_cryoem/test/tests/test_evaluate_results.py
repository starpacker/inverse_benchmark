import sys
import os
import dill
import traceback
import numpy as np

# Add path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze paths to determine scenario
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
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run function
    try:
        print("\n[Phase 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Number of args: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("\n[Phase 2] Executing evaluate_results...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("  Function executed successfully")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n[Scenario B] Factory/Closure pattern detected")
        
        if not callable(result):
            print(f"ERROR: Expected callable result for closure pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"\n  Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"  Inner function: {inner_data.get('func_name', 'unknown')}")
                print(f"  Inner args count: {len(inner_args)}")
                
                # Execute the inner operator
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"\nTEST FAILED for inner path {inner_path}")
                    print(f"Mismatch details: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed for: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed processing inner path {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function
        print("\n[Scenario A] Simple function pattern detected")
        
        expected = outer_output
        
        # Verify
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"\nTEST FAILED")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
            else:
                print("\nTEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()