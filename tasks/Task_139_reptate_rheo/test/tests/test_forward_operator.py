import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/reptate_rheo_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    # ---- File Logic Setup ----
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct operator ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
        print(f"[INFO] Function name in data: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # ---- Phase 2: Execution & Verification ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print(f"[INFO] forward_operator returned type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error calling forward_operator with outer args: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: forward_operator did not return a callable, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"[INFO] Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple Function
        print("[INFO] Scenario A detected: Simple function call")
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print(f"[INFO] forward_operator returned type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error calling forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed")
            print(f"  Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()