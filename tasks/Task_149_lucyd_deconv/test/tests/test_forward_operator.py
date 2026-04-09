import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/lucyd_deconv_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Loaded outer data successfully.")
    except Exception as e:
        print("ERROR: Failed to load outer data: {}".format(str(e)))
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print("Phase 1: Operator created successfully.")
        except Exception as e:
            print("ERROR: Failed to create operator: {}".format(str(e)))
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: The returned operator is not callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Loaded inner data: {}".format(os.path.basename(inner_path)))
            except Exception as e:
                print("ERROR: Failed to load inner data: {}".format(str(e)))
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print("ERROR: Failed to execute operator: {}".format(str(e)))
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print("TEST FAILED: {}".format(msg))
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print("ERROR: Verification failed: {}".format(str(e)))
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print("Function executed successfully.")
        except Exception as e:
            print("ERROR: Failed to execute function: {}".format(str(e)))
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print("TEST FAILED: {}".format(msg))
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print("ERROR: Verification failed: {}".format(str(e)))
            traceback.print_exc()
            sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()