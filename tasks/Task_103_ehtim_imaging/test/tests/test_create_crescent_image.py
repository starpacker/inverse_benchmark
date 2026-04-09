import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_crescent_image import create_crescent_image
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/ehtim_imaging_sandbox_sandbox/run_code/std_data/standard_data_create_crescent_image.pkl'
    ]

    # Step 1: Classify data files into outer (direct) and inner (parent/closure) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_create_crescent_image.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_crescent_image.pkl).")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Step 3: Phase 1 - Run the function with outer args
    try:
        agent_result = create_crescent_image(*outer_args, **outer_kwargs)
        print("Phase 1: create_crescent_image executed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to execute create_crescent_image: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print("ERROR: Expected agent_result to be callable for Scenario B, but it is not.")
            print(f"  Type of agent_result: {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"Phase 2 [{idx}]: Operator executed successfully.")
            except Exception as e:
                print(f"ERROR: Failed to execute operator [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"PASSED [{idx}]")

        if not all_passed:
            print("TEST FAILED")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function - compare result directly
        print("Scenario A detected: Simple function call.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAILED: {msg}")
            print("TEST FAILED")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()