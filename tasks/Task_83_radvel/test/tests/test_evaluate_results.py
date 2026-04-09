import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/radvel_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
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
    
    # Phase 1: Load outer data and run function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute function based on scenario
    try:
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print("\nScenario B: Factory/Closure pattern detected")
            print("Running outer function to get operator...")
            
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            # Load inner data
            inner_path = inner_paths[0]
            print(f"\nLoading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print("Executing operator with inner arguments...")
            result = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple function
            print("\nScenario A: Simple function pattern detected")
            print("Running evaluate_results directly...")
            
            result = evaluate_results(*outer_args, **outer_kwargs)
        
        print(f"\nResult type: {type(result)}")
        print(f"Expected type: {type(expected_output)}")
        
    except Exception as e:
        print(f"ERROR during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Verification
    try:
        print("\nVerifying results...")
        passed, msg = recursive_check(expected_output, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print(f"Mismatch details: {msg}")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()