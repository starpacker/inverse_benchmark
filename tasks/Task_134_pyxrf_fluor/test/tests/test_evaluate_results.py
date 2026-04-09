import sys
import os
import dill
import traceback
import numpy as np
import tempfile
import shutil

# Ensure matplotlib doesn't try to open windows
import matplotlib
matplotlib.use('Agg')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pyxrf_fluor_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # The function writes files to results_dir. We need to use a temp directory
    # to avoid polluting the filesystem, but we must match the expected output.
    # We'll redirect results_dir to a temp directory.
    # results_dir is the 5th positional arg (index 4) or keyword 'results_dir'
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')

        # Modify results_dir to use temp directory
        args_list = list(outer_args)
        if len(args_list) > 4:
            args_list[4] = temp_dir
        elif 'results_dir' in outer_kwargs:
            outer_kwargs['results_dir'] = temp_dir
        else:
            args_list[4] = temp_dir

        outer_args = tuple(args_list)

        # Phase 1: Execute the function
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("[INFO] evaluate_results executed successfully.")
        except Exception as e:
            print(f"FAIL: evaluate_results raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Check if this is Scenario B (factory pattern)
        if inner_paths:
            # Scenario B: result should be callable
            if not callable(result):
                print("FAIL: Expected callable from evaluate_results (factory pattern), got non-callable.")
                sys.exit(1)

            agent_operator = result

            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"[INFO] Loaded inner data from: {inner_path}")
                except Exception as e:
                    print(f"FAIL: Could not load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)

                try:
                    inner_result = agent_operator(*inner_args, **inner_kwargs)
                    print("[INFO] Inner operator executed successfully.")
                except Exception as e:
                    print(f"FAIL: Inner operator raised an exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                try:
                    passed, msg = recursive_check(inner_expected, inner_result)
                    if not passed:
                        print(f"FAIL: Verification failed for inner call: {msg}")
                        sys.exit(1)
                    else:
                        print(f"[PASS] Inner call verification passed.")
                except Exception as e:
                    print(f"FAIL: Verification raised an exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)

        else:
            # Scenario A: Direct comparison
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("[PASS] Direct output verification passed.")
            except Exception as e:
                print(f"FAIL: Verification raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()