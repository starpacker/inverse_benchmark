import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path if needed
sys.path.insert(0, '/data/yjh/bilby_gw_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/bilby_gw_sandbox_sandbox')

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/bilby_gw_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

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
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print("Phase 1: forward_operator executed, got operator.")
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: forward_operator did not return a callable.")
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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: operator executed successfully.")
            except Exception as e:
                print(f"FAIL: operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print("forward_operator executed successfully.")
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

if __name__ == '__main__':
    main()