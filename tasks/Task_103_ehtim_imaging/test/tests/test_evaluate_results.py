import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/ehtim_imaging_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # ---- Phase 1: Load outer data and run evaluate_results ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        from agent_evaluate_results import evaluate_results
        print("[INFO] Imported evaluate_results successfully.")
    except Exception as e:
        print(f"FAIL: Could not import evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        agent_result = evaluate_results(*outer_args, **outer_kwargs)
        print("[INFO] evaluate_results executed successfully.")
    except Exception as e:
        print(f"FAIL: evaluate_results raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Phase 2: Determine scenario and verify ----
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected (inner data files found).")
        if not callable(agent_result):
            print(f"FAIL: Expected callable from evaluate_results, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner call executed successfully for {os.path.basename(ip)}.")
            except Exception as e:
                print(f"FAIL: Inner call raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL [{os.path.basename(ip)}]: {msg}")
                all_passed = False
            else:
                print(f"[PASS] Inner test passed for {os.path.basename(ip)}")

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected (simple function).")
        result = agent_result
        expected = outer_output

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


if __name__ == '__main__':
    main()