import sys
import os
import dill
import torch
import numpy as np
import traceback

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/phiflow_pde_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
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

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print("Loaded outer data successfully.")
    except Exception as e:
        print("ERROR loading outer data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print("Phase 1: forward_operator returned successfully.")
        except Exception as e:
            print("ERROR in Phase 1 (creating operator): " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: forward_operator did not return a callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print("Loaded inner data: " + os.path.basename(inner_path))
            except Exception as e:
                print("ERROR loading inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print("ERROR executing inner operator: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print("TEST FAILED: " + str(msg))
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
    else:
        # Scenario A: Simple function
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print("Phase 1: forward_operator executed successfully.")
        except Exception as e:
            print("ERROR executing forward_operator: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output
        passed, msg = recursive_check(expected, result)
        if not passed:
            print("TEST FAILED: " + str(msg))
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

if __name__ == '__main__':
    main()