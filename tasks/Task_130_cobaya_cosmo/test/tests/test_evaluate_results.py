import sys
import os
import dill
import traceback
import numpy as np

# Ensure the data directory and current directory are on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/data/yjh/cobaya_cosmo_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/cobaya_cosmo_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Separate outer vs inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # ---- Phase 1: Load outer data ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # ---- Determine scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        # Run outer function to get the operator/callable
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Outer function executed successfully.")
        except Exception as e:
            print(f"FAIL: Outer function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer function, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner function executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function")

        # Use a temporary directory for results_dir to avoid polluting original location
        import tempfile
        temp_results_dir = None

        # Check if results_dir argument exists and override it with a temp dir
        # so we don't fail on permission issues or overwrite real results
        try:
            # Modify results_dir to use a temp directory
            args_list = list(outer_args)
            # results_dir is the 4th positional argument (index 3)
            if len(args_list) >= 4:
                temp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
                args_list[3] = temp_results_dir
                outer_args = tuple(args_list)
            elif 'results_dir' in outer_kwargs:
                temp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
                outer_kwargs['results_dir'] = temp_results_dir
        except Exception:
            pass  # If modification fails, proceed with original args

        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Function executed successfully.")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Clean up temp directory
        if temp_results_dir and os.path.exists(temp_results_dir):
            import shutil
            try:
                shutil.rmtree(temp_results_dir)
            except Exception:
                pass

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()