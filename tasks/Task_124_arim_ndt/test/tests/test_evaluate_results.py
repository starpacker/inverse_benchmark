import sys
import os
import dill
import traceback
import numpy as np

# Ensure the results directory exists for file-saving operations in the function
# We need to handle this before running the function

def main():
    data_paths = ['/data/yjh/arim_ndt_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

    # Step 1: Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Step 3: Ensure results_dir exists (the function writes files there)
    try:
        # data_dict is the first argument
        if len(outer_args) > 0 and isinstance(outer_args[0], dict):
            results_dir = outer_args[0].get('results_dir', None)
            if results_dir is not None:
                os.makedirs(results_dir, exist_ok=True)
                print(f"  Ensured results_dir exists: {results_dir}")
    except Exception as e:
        print(f"Warning: Could not create results_dir: {e}")

    # Step 4: Import the target function
    try:
        from agent_evaluate_results import evaluate_results
        print("Successfully imported evaluate_results")
    except Exception as e:
        print(f"FAIL: Could not import evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Import verification utility
    try:
        from verification_utils import recursive_check
        print("Successfully imported recursive_check")
    except Exception as e:
        print(f"FAIL: Could not import recursive_check: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")

        # Phase 1: Create operator
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Returned operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED: {msg}")
                else:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"FAIL: Comparison error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function call")

        # Phase 1: Execute function
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully executed evaluate_results")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print(f"TEST PASSED: {msg}")
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Comparison error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()