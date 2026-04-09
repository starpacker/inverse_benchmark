import sys
import os
import dill
import numpy as np
import traceback

from agent_create_cartesian_mask import create_cartesian_mask
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data_create_cartesian_mask.pkl']

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        agent_result = create_cartesian_mask(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR executing create_cartesian_mask: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable from create_cartesian_mask but got non-callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for inner path: {inner_path}")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: The result from Phase 1 IS the result
        result = agent_result
        expected = expected_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()