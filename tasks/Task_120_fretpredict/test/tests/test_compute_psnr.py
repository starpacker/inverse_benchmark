import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_psnr import compute_psnr

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_compute_psnr.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for compute_psnr.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}")
        print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute function
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = compute_psnr(*outer_args, **outer_kwargs)
            print(f"Phase 1: compute_psnr returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error calling compute_psnr with outer args: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from compute_psnr, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: agent_operator returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error calling agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Verification
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner test passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        try:
            result = compute_psnr(*outer_args, **outer_kwargs)
            print(f"Phase 1: compute_psnr returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error calling compute_psnr: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Verification
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
                print(f"  Message:  {msg}")
                sys.exit(1)
            else:
                print("PASS: Result matches expected output.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()