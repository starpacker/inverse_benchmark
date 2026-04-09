import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_norm import compute_norm

# Import verification utility
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/gsas2_rietveld_sandbox_sandbox/run_code/std_data/standard_data_compute_norm.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_norm.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_compute_norm.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Loaded outer data: func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_result = compute_norm(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: compute_norm raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected compute_norm to return a callable, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            print(f"Processing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"  Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                print(f"  Expected: {inner_expected}")
                print(f"  Actual:   {result}")
                sys.exit(1)
            else:
                print(f"  Inner test passed: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function
        print("Detected Scenario A (Simple function)")

        result = agent_result

        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            print(f"  Expected: {expected_output}")
            print(f"  Actual:   {result}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()