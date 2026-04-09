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
    data_paths = ['/data/yjh/NLOS_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Analyze data paths to determine test strategy
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
        print(f"  - Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\nDetected Scenario B: Factory/Closure Pattern")
        
        try:
            # Execute outer function to get the operator
            print("Executing evaluate_results to get operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"WARNING: Result is not callable, treating as Scenario A")
                # Fall back to Scenario A
                result = agent_operator
                expected = outer_output
            else:
                print(f"Operator obtained successfully: {type(agent_operator)}")
                
                # Load inner data
                inner_path = inner_paths[0]  # Use first inner path
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - Number of args: {len(inner_args)}")
                print(f"  - Kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute operator with inner data
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
        except Exception as e:
            print(f"ERROR: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\nDetected Scenario A: Simple Function")
        
        try:
            print("Executing evaluate_results...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            expected = outer_output
            print("Function executed successfully.")
            
        except Exception as e:
            print(f"ERROR: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 2: Verification
    print("\nVerifying results...")
    try:
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()