import sys
import os
import dill
import traceback

# Import the target function
from agent__log_likelihood import _log_likelihood
from verification_utils import recursive_check

def main():
    """Main test function for _log_likelihood."""
    
    # Data paths provided
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data__log_likelihood.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__log_likelihood.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        print(f"[INFO] Scenario A: Simple function test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args, kwargs, and expected output
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"[INFO] Outer args count: {len(outer_args)}")
        print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the function
        try:
            print(f"[INFO] Executing _log_likelihood with provided arguments...")
            actual_result = _log_likelihood(*outer_args, **outer_kwargs)
            print(f"[INFO] Execution completed successfully")
        except Exception as e:
            print(f"[ERROR] Failed to execute _log_likelihood: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            print(f"[INFO] Comparing results...")
            passed, msg = recursive_check(expected_output, actual_result)
            
            if passed:
                print(f"[INFO] TEST PASSED")
                sys.exit(0)
            else:
                print(f"[ERROR] TEST FAILED: {msg}")
                print(f"[DEBUG] Expected type: {type(expected_output)}")
                print(f"[DEBUG] Actual type: {type(actual_result)}")
                if hasattr(expected_output, 'shape'):
                    print(f"[DEBUG] Expected shape: {expected_output.shape}")
                if hasattr(actual_result, 'shape'):
                    print(f"[DEBUG] Actual shape: {actual_result.shape}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (has inner paths)
    elif outer_path and inner_paths:
        print(f"[INFO] Scenario B: Factory/Closure pattern test")
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract outer args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[INFO] Outer args count: {len(outer_args)}")
        print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Phase 1: Create the operator/closure
        try:
            print(f"[INFO] Creating operator via _log_likelihood...")
            agent_operator = _log_likelihood(*outer_args, **outer_kwargs)
            print(f"[INFO] Operator created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"[ERROR] Created operator is not callable. Type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print(f"[INFO] Inner args count: {len(inner_args)}")
            print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            try:
                print(f"[INFO] Executing operator with inner arguments...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Execution completed successfully")
            except Exception as e:
                print(f"[ERROR] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                print(f"[INFO] Comparing results...")
                passed, msg = recursive_check(expected_output, actual_result)
                
                if passed:
                    print(f"[INFO] Test for {os.path.basename(inner_path)} PASSED")
                else:
                    print(f"[ERROR] TEST FAILED: {msg}")
                    print(f"[DEBUG] Expected type: {type(expected_output)}")
                    print(f"[DEBUG] Actual type: {type(actual_result)}")
                    sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print(f"[INFO] TEST PASSED")
        sys.exit(0)
    
    else:
        print(f"[ERROR] Could not determine test scenario from data paths")
        print(f"[DEBUG] Data paths: {data_paths}")
        print(f"[DEBUG] Outer path: {outer_path}")
        print(f"[DEBUG] Inner paths: {inner_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()