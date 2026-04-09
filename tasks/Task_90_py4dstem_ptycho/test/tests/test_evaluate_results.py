import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
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
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully")
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args count: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        
        try:
            # Execute outer function to get the operator/closure
            print("Executing outer function to get operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"Warning: Result is not callable, treating as Scenario A")
                # Fall back to Scenario A
                result = agent_operator
                expected = outer_output
            else:
                print(f"Operator obtained: {type(agent_operator)}")
                
                # Load inner data
                inner_path = inner_paths[0]
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully")
                print(f"  Function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Args count: {len(inner_args)}")
                print(f"  Kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("\nExecuting operator with inner arguments...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
        except Exception as e:
            print(f"ERROR during execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function")
        
        try:
            print("Executing function...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            expected = outer_output
            
        except Exception as e:
            print(f"ERROR during execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 2: Verification
    print("\n" + "="*50)
    print("VERIFICATION PHASE")
    print("="*50)
    
    try:
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        if isinstance(expected, dict):
            print(f"Expected keys: {list(expected.keys())}")
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print("="*50)
            print(f"Mismatch details: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()