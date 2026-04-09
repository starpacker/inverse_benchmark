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
        '/data/yjh/straintool_geo_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # ── Step 1: Classify files ──────────────────────────────────────────
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found among provided paths.")
        sys.exit(1)

    # ── Step 2: Load outer data ─────────────────────────────────────────
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name : {outer_data.get('func_name')}")
        print(f"  args len  : {len(outer_data.get('args', []))}")
        print(f"  kwargs keys: {list(outer_data.get('kwargs', {}).keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # ── Step 3: Execute the target function ─────────────────────────────
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 4: Determine scenario & set expected / actual ──────────────
    if inner_paths:
        # ── Scenario B: Factory / Closure pattern ───────────────────────
        print("Detected Scenario B (factory/closure pattern).")

        if not callable(agent_result):
            print("FAIL: Expected a callable operator from outer call, "
                  f"got {type(agent_result)}")
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
            expected = inner_data.get('output')

            try:
                actual = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL (inner data): {msg}")
                sys.exit(1)

            print(f"  Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # ── Scenario A: Simple function ─────────────────────────────────
        print("Detected Scenario A (simple function).")

        expected = outer_data.get('output')
        actual = agent_result

        try:
            passed, msg = recursive_check(expected, actual)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()