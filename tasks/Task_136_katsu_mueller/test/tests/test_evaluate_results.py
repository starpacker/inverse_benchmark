import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/katsu_mueller_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Override results_dir to a temp directory to avoid polluting the original
    # Check if results_dir is in kwargs or args
    import tempfile
    tmp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')

    # The function signature is evaluate_results(M_true, M_recon, results_dir=None)
    # We need to override results_dir to avoid writing to the original location
    # and to avoid path issues with __file__ resolution
    modified_kwargs = dict(outer_kwargs)
    # If results_dir is passed as a positional arg (3rd), handle it
    if len(outer_args) >= 3:
        outer_args = list(outer_args)
        outer_args[2] = tmp_results_dir
        outer_args = tuple(outer_args)
    else:
        modified_kwargs['results_dir'] = tmp_results_dir

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        try:
            agent_operator = evaluate_results(*outer_args, **modified_kwargs)
            print(f"[INFO] Phase 1 complete. agent_operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Phase 1 (operator creation) failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Phase 2 execution complete. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Phase 2 (operator execution) failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"[INFO] Inner path verified: {inner_path}")

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: Simple function")

        try:
            result = evaluate_results(*outer_args, **modified_kwargs)
            print(f"[INFO] Function execution complete. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(tmp_results_dir, ignore_errors=True)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()