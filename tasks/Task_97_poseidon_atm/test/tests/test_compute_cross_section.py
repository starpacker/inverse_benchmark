import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_cross_section import compute_cross_section

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/poseidon_atm_sandbox_sandbox/run_code/std_data/standard_data_compute_cross_section.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_cross_section.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_compute_cross_section.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator / compute result
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

    try:
        agent_result = compute_cross_section(*outer_args, **outer_kwargs)
        print("Phase 1: compute_cross_section executed successfully.")
    except Exception as e:
        print(f"FAIL: compute_cross_section raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from compute_cross_section, got {type(agent_result)}")
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
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print("Phase 2: Inner execution completed successfully.")
            except Exception as e:
                print(f"FAIL: Inner execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"  Inner test passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function verification.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()