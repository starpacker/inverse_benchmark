import sys
import os
import dill
import traceback
import numpy as np

# Import target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/flowdec_deconv_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Classify paths
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

    # Phase 1: Load outer data and reconstruct
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: load_and_preprocess_data execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_operator):
            print("FAIL: Expected callable operator from outer call, got non-callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution failed on {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                print(f"Details: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')
        result = agent_operator

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Details: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()