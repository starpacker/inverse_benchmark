import sys
import os
import dill
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/NoisePy_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Scenario A: Simple function (only outer data exists)
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl")
        sys.exit(1)
    
    # Load outer data
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
    outer_output = outer_data.get('output')
    
    # Check if we have inner paths (Scenario B: Factory/Closure Pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory pattern
        print("Detected factory/closure pattern (Scenario B)")
        
        # Phase 1: Run outer function to get the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created agent operator")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
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
            expected = inner_data.get('output')
            
            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed agent operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern (Scenario A)")
        
        # Execute the function
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully executed evaluate_results")
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()