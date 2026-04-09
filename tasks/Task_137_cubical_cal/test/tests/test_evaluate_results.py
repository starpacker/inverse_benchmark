import sys
import os
import dill
import traceback
import numpy as np

# Ensure the module directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/cubical_cal_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

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

    # Phase 1: Load outer data
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

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results returned an operator/callable.")
        except Exception as e:
            print(f"FAIL: evaluate_results raised an exception during Phase 1: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Expected callable from evaluate_results in Scenario B, got non-callable.")
            sys.exit(1)

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
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Operator execution raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - result is the direct output
        try:
            # The function writes files to results_dir. We may need to use a temp dir
            # to avoid issues, but let's use the args as-is first.
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results executed successfully.")
        except Exception as e:
            print(f"FAIL: evaluate_results raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()