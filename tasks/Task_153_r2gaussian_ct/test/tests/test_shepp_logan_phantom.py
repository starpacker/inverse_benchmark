import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_shepp_logan_phantom import shepp_logan_phantom

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/r2gaussian_ct_sandbox_sandbox/run_code/std_data/standard_data_shepp_logan_phantom.pkl'
    ]

    # Step 1: Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_shepp_logan_phantom.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
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

    # Step 3: Execute the function
    try:
        agent_result = shepp_logan_phantom(*outer_args, **outer_kwargs)
        print("Successfully executed shepp_logan_phantom(*outer_args, **outer_kwargs)")
    except Exception as e:
        print(f"FAIL: Error executing shepp_logan_phantom: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from shepp_logan_phantom, got {type(agent_result)}")
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
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed agent_result(*inner_args, **inner_kwargs)")
            except Exception as e:
                print(f"FAIL: Error executing inner call: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Step 5: Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Details: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: No inner data files. Comparing direct output.")

        result = agent_result
        expected = outer_data.get('output')

        # Step 5: Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Details: {msg}")
                sys.exit(1)
            else:
                print("PASS: Output verified successfully.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()