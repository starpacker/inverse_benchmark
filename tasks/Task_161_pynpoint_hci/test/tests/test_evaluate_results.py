import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        # Phase 1: Create the operator
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] evaluate_results returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: evaluate_results raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
                print(f"[INFO] Operator execution returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner test passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call")

        # Phase 1: Execute the function
        try:
            # Remove save_dir and vis_path to avoid side effects during testing
            # but only if they were in the original kwargs
            test_kwargs = dict(outer_kwargs)
            # We keep the original args/kwargs as-is since the data was captured that way
            result = evaluate_results(*outer_args, **test_kwargs)
            print(f"[INFO] evaluate_results returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: evaluate_results raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Phase 2: Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                # Print additional debug info
                if isinstance(expected, dict) and isinstance(result, dict):
                    print(f"[DEBUG] Expected keys: {list(expected.keys())}")
                    print(f"[DEBUG] Result keys: {list(result.keys())}")
                    for key in expected:
                        if key in result:
                            try:
                                k_passed, k_msg = recursive_check(expected[key], result[key])
                                if not k_passed:
                                    print(f"[DEBUG] Key '{key}' mismatch: {k_msg}")
                                    print(f"[DEBUG]   Expected: {expected[key]}")
                                    print(f"[DEBUG]   Got:      {result[key]}")
                            except Exception:
                                pass
                        else:
                            print(f"[DEBUG] Key '{key}' missing from result")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()