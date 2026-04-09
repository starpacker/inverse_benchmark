import sys
import os
import dill
import numpy as np
import traceback

from agent_divergence_2d import divergence_2d
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data_divergence_2d.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_divergence_2d.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find standard_data_divergence_2d.pkl")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = divergence_2d(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: divergence_2d did not return a callable operator")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED for {inner_path}: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function
        try:
            result = divergence_2d(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR executing divergence_2d: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()