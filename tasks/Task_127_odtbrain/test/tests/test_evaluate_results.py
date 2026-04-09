import sys
import os
import dill
import traceback
import numpy as np

# Ensure the necessary paths are available
sys.path.insert(0, '/data/yjh/odtbrain_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = ['/data/yjh/odtbrain_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

    # Separate outer and inner paths
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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # The function writes files to results_dir. We need to ensure that directory exists
    # and potentially use a temp directory to avoid conflicts. However, we use the same
    # args as captured to maintain consistency.
    # Make sure results_dir from args exists
    try:
        # results_dir is the 5th positional argument (index 4)
        if len(outer_args) >= 5:
            results_dir = outer_args[4]
            os.makedirs(results_dir, exist_ok=True)
        elif 'results_dir' in outer_kwargs:
            results_dir = outer_kwargs['results_dir']
            os.makedirs(results_dir, exist_ok=True)
    except Exception:
        pass

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        # Phase 1: Create the operator
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results executed successfully (operator created).")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("evaluate_results executed successfully.")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()