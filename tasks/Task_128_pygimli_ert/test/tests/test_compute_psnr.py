import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_psnr import compute_psnr
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data/standard_data_compute_psnr.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_psnr.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file 'standard_data_compute_psnr.pkl'")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Loaded outer data: func_name={outer_data.get('func_name')}, "
              f"args count={len(outer_args)}, kwargs keys={list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Run outer function to get operator
        try:
            agent_operator = compute_psnr(*outer_args, **outer_kwargs)
            print(f"Outer function returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error running outer function compute_psnr: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer function, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"  Inner args count={len(inner_args)}, kwargs keys={list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function call")

        try:
            result = compute_psnr(*outer_args, **outer_kwargs)
            print(f"Function returned: {type(result)} = {result}")
        except Exception as e:
            print(f"FAIL: Error running compute_psnr: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Verification passed")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()