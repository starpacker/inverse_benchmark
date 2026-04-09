import sys
import os
import dill
import traceback
import numpy as np

# Ensure matplotlib doesn't try to open display
import matplotlib
matplotlib.use('Agg')

def main():
    data_paths = [
        '/data/yjh/21cmfast_tomo_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Step 1: Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found for evaluate_results.")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Import the target function
    try:
        from agent_evaluate_results import evaluate_results
        print("Successfully imported evaluate_results")
    except Exception as e:
        print(f"ERROR importing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Import verification utility
    try:
        from verification_utils import recursive_check
        print("Successfully imported recursive_check")
    except Exception as e:
        print(f"ERROR importing recursive_check: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")
        print(f"  Found {len(inner_paths)} inner data file(s)")

        # Phase 1: Create the operator
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator from evaluate_results")
        except Exception as e:
            print(f"ERROR in Phase 1 (creating operator): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from: {inner_path}")
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner data")
            except Exception as e:
                print(f"ERROR in Phase 2 (executing operator): {e}")
                traceback.print_exc()
                sys.exit(1)

            # Step 6: Compare
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for inner data: {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for inner data: {os.path.basename(inner_path)}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function call")

        # Modify results_dir to use a temp directory to avoid conflicts
        # The function writes files; we need to ensure the directory is writable
        # We'll use the same args but potentially override results_dir if needed
        try:
            # Use a temporary results directory to avoid overwriting original results
            import tempfile
            temp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
            print(f"Using temporary results directory: {temp_results_dir}")

            # Check if results_dir is in args or kwargs and override it
            # Based on the function signature, results_dir is the 11th positional arg (index 10)
            modified_args = list(outer_args)
            if 'results_dir' in outer_kwargs:
                outer_kwargs['results_dir'] = temp_results_dir
            elif len(modified_args) >= 11:
                modified_args[10] = temp_results_dir
            modified_args = tuple(modified_args)

            result = evaluate_results(*modified_args, **outer_kwargs)
            print("Successfully executed evaluate_results")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Step 6: Compare
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Clean up temporary directory if created
    try:
        if 'temp_results_dir' in dir() and os.path.exists(temp_results_dir):
            import shutil
            shutil.rmtree(temp_results_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {temp_results_dir}")
    except Exception:
        pass

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()