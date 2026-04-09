import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_generate_scan_positions import generate_scan_positions

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/ptychi_recon_sandbox_sandbox/run_code/std_data/standard_data_generate_scan_positions.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_generate_scan_positions.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_scan_positions.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator / get result
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Loaded outer data: func_name={outer_data.get('func_name')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_result = generate_scan_positions(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute generate_scan_positions with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        if not callable(agent_result):
            print(f"ERROR: Expected callable from generate_scan_positions, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            print(f"Processing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED for inner data {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call")

        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            # Print some debug info
            print(f"  Expected type: {type(expected)}")
            print(f"  Result type: {type(result)}")
            if isinstance(expected, list) and isinstance(result, list):
                print(f"  Expected length: {len(expected)}, Result length: {len(result)}")
                if len(expected) > 0 and len(result) > 0:
                    print(f"  Expected[0]: {expected[0]}")
                    print(f"  Result[0]: {result[0]}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()