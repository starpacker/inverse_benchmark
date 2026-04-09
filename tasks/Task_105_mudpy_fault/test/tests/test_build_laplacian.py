import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_build_laplacian import build_laplacian

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/data/yjh/mudpy_fault_sandbox_sandbox/run_code/std_data/standard_data_build_laplacian.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_build_laplacian.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_build_laplacian.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Loaded outer data: func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_result = build_laplacian(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: Error executing build_laplacian(*outer_args, **outer_kwargs): {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected agent_result to be callable for Scenario B, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Running inner call from {os.path.basename(inner_path)}...")

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_result(*inner_args, **inner_kwargs): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Inner test from {os.path.basename(inner_path)} failed: {msg}")
                all_passed = False
            else:
                print(f"  Inner test from {os.path.basename(inner_path)} passed.")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call.")

        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Test failed: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()