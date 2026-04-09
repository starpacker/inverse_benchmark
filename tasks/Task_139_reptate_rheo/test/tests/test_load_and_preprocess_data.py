import sys
import os
import dill
import traceback
import numpy as np

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/reptate_rheo_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Separate outer and inner paths
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

    # Phase 1: Load outer data
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

    # Phase 1: Run the function
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"[INFO] Function executed successfully. Result type: {type(agent_result)}")
    except Exception as e:
        print(f"FAIL: Error executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from outer function, got {type(agent_result)}")
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
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner function executed successfully. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error executing inner function: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner verification passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: Simple function")

        result = agent_result
        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Verification passed.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()