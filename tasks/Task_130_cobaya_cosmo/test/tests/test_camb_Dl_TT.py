import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_camb_Dl_TT import camb_Dl_TT
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/cobaya_cosmo_sandbox_sandbox/run_code/std_data/standard_data_camb_Dl_TT.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_camb_Dl_TT.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_camb_Dl_TT.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # Phase 1: Create operator
        try:
            agent_operator = camb_Dl_TT(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure.")
        except Exception as e:
            print(f"FAIL: Phase 1 - Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Phase 1 - Result is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
                print("Phase 2: Successfully executed operator with inner args.")
            except Exception as e:
                print(f"FAIL: Phase 2 - Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        # Execute the function directly and compare with expected output
        try:
            result = camb_Dl_TT(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully executed camb_Dl_TT.")
        except Exception as e:
            print(f"FAIL: Could not execute camb_Dl_TT: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()