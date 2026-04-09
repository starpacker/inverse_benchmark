import sys
import os
import dill
import numpy as np
import traceback

# Import target function and verification utility
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/flowdec_deconv_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    # Classify paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data and run function
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
        agent_result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: forward_operator execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_result):
            print("FAIL: Expected callable from forward_operator, got " + type(agent_result).__name__)
            sys.exit(1)

        for inner_path in sorted(inner_paths):
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
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Inner call failed for {os.path.basename(inner_path)}: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL [{os.path.basename(inner_path)}]: {msg}")
                sys.exit(1)
            print(f"PASS [{os.path.basename(inner_path)}]")
    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')
        result = agent_result

        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()