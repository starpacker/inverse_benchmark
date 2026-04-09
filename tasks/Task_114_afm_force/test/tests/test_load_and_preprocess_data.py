import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/afm_force_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # ── Step 1: Classify data files ──────────────────────────────────────
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer (standard_data_load_and_preprocess_data.pkl) data file.")
        sys.exit(1)

    # ── Step 2: Load outer data ──────────────────────────────────────────
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data loaded successfully from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Execute the function with outer args ─────────────────────
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 4: Determine scenario and verify ────────────────────────────
    if inner_paths:
        # ── Scenario B: Factory / Closure pattern ────────────────────────
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s).")

        # The outer call should have returned a callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from outer call, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result
        all_passed = True

        for idx, ip in enumerate(inner_paths):
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                print(f"  Inner data [{idx}] loaded from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception for inner [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_output, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner [{idx}]: {msg}")
                    all_passed = False
                else:
                    print(f"  Inner [{idx}] verification PASSED.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception for inner [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        # ── Scenario A: Simple function call ─────────────────────────────
        print("Scenario A detected: Simple function call.")

        result = agent_result
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
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()