import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_sart_tv import sart_tv
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/r2gaussian_ct_sandbox_sandbox/run_code/std_data/standard_data_sart_tv.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_sart_tv.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find standard_data_sart_tv.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = sart_tv(*outer_args, **outer_kwargs)
            print("Phase 1: sart_tv executed, got operator.")
        except Exception as e:
            print(f"ERROR executing sart_tv (Phase 1): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute inner calls
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"ERROR executing operator (Phase 2): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        expected = outer_data.get('output')

        try:
            result = sart_tv(*outer_args, **outer_kwargs)
            print("sart_tv executed successfully.")
        except Exception as e:
            print(f"ERROR executing sart_tv: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()