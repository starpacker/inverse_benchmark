import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
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
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Phase 2: Execute and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Phase 1: Reconstructed operator successfully.")
        except Exception as e:
            print(f"FAIL: Could not reconstruct operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Reconstructed operator is not callable (expected closure/factory pattern).")
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
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Executed operator successfully.")
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Verification succeeded for inner path {inner_path}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        try:
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Phase 1: Executed function successfully.")
        except Exception as e:
            print(f"FAIL: Could not execute function: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("PASS: Verification succeeded.")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()