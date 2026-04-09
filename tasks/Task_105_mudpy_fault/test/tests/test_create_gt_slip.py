import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_gt_slip import create_gt_slip
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/mudpy_fault_sandbox_sandbox/run_code/std_data/standard_data_create_gt_slip.pkl']

    # Step 1: Classify data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_create_gt_slip.pkl)")
        sys.exit(1)

    # Step 2: Load outer data
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

    # Step 3: Execute the function (Phase 1)
    try:
        agent_result = create_gt_slip(*outer_args, **outer_kwargs)
        print("Successfully called create_gt_slip with outer args/kwargs.")
    except Exception as e:
        print(f"FAIL: Error executing create_gt_slip: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from create_gt_slip, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
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
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed the returned operator with inner args/kwargs.")
            except Exception as e:
                print(f"FAIL: Error executing returned operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL (inner): {msg}")
                    all_passed = False
                else:
                    print(f"Inner test passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()