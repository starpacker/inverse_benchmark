import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/impedance_eis_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct operator ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Successfully called load_and_preprocess_data with outer args/kwargs.")
    except Exception as e:
        print(f"FAIL: Error executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Phase 2: Determine scenario and verify ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern with inner data files.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from load_and_preprocess_data, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print("[INFO] Successfully executed inner operator call.")
            except Exception as e:
                print(f"FAIL: Error executing inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"[INFO] Inner data {os.path.basename(inner_path)} passed verification.")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        print("[INFO] Scenario A detected: Simple function call.")

        expected = outer_output
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()