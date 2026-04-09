import sys
import os
import dill
import numpy as np
import traceback

from agent_fwhm_to_sigma import fwhm_to_sigma
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pyxrf_fluor_sandbox_sandbox/run_code/std_data/standard_data_fwhm_to_sigma.pkl']

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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = fwhm_to_sigma(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Result of fwhm_to_sigma is not callable but inner data exists.")
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function
        try:
            result = fwhm_to_sigma(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not execute fwhm_to_sigma: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()