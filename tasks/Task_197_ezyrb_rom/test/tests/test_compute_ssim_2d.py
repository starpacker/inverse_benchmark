import sys
import os
import dill
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_compute_ssim_2d import compute_ssim_2d
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_compute_ssim_2d.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Loaded outer data successfully.")
    except Exception as e:
        print("ERROR loading outer data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = compute_ssim_2d(*outer_args, **outer_kwargs)
            print("Phase 1: Created operator successfully.")
        except Exception as e:
            print("ERROR in Phase 1 (creating operator): " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable.")
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print("Loaded inner data: " + os.path.basename(ip))
            except Exception as e:
                print("ERROR loading inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Executed operator successfully.")
            except Exception as e:
                print("ERROR in Phase 2 (executing operator): " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print("ERROR during verification: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print("TEST FAILED: " + str(msg))
                sys.exit(1)

    else:
        # Scenario A: Simple function
        try:
            result = compute_ssim_2d(*outer_args, **outer_kwargs)
            print("Executed compute_ssim_2d successfully.")
        except Exception as e:
            print("ERROR executing compute_ssim_2d: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print("ERROR during verification: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print("TEST FAILED: " + str(msg))
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()