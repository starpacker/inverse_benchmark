import sys
import os
import dill
import numpy as np
import traceback

# Add the necessary paths
sys.path.insert(0, '/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code/std_data')

from agent_assemble import assemble
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code/std_data/standard_data_assemble.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_assemble.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_assemble.pkl in data_paths.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer args types: {[type(a).__name__ for a in outer_args]}")
        print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute assemble
    try:
        agent_result = assemble(*outer_args, **outer_kwargs)
        print(f"[INFO] assemble() returned type: {type(agent_result).__name__}")
    except Exception as e:
        print(f"FAIL: assemble() raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 3: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected with {len(inner_paths)} inner data file(s).")

        # Verify agent_result is callable
        if not callable(agent_result):
            print("FAIL: Scenario B expected callable from assemble(), but got non-callable.")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Executing operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"[INFO] Verification passed for {inner_path}")
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
        print("[INFO] Scenario A detected (simple function).")

        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
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