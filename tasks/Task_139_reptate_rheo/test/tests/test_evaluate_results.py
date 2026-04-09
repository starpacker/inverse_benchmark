import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/reptate_rheo_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("ERROR: No outer data file found (standard_data_evaluate_results.pkl).")
        sys.exit(1)

    # ── Phase 1: Load outer data ──
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # If results_dir is among args, we may want to redirect it to a temp dir
    # to avoid overwriting original results. The function writes files to results_dir.
    # We'll use a temporary directory to avoid side effects.
    import tempfile
    temp_results_dir = tempfile.mkdtemp(prefix="test_evaluate_results_")

    # The results_dir is the last positional argument (index 9) based on the function signature.
    # We need to replace it so the test doesn't write to the original location.
    outer_args_list = list(outer_args)
    if len(outer_args_list) >= 10:
        # Replace results_dir (last positional arg) with temp dir
        outer_args_list[9] = temp_results_dir
    elif 'results_dir' in outer_kwargs:
        outer_kwargs['results_dir'] = temp_results_dir
    else:
        # If fewer args, try to append or set as kwarg
        outer_kwargs['results_dir'] = temp_results_dir
    outer_args = tuple(outer_args_list)

    # ── Determine scenario ──
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern.")

        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] evaluate_results returned: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR executing evaluate_results (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner call returned: {type(result)}")
            except Exception as e:
                print(f"ERROR executing inner call: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call.")

        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] evaluate_results returned: {type(result)}")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()