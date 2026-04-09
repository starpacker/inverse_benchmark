import sys
import os
import dill
import traceback
import numpy as np

# Ensure the target module's directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pyroomacoustics_doa_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # -------------------------------------------------------------------
    # Step 1: Classify files into outer (simple/factory) and inner (closure)
    # -------------------------------------------------------------------
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl).")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 2: Load outer data
    # -------------------------------------------------------------------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # -------------------------------------------------------------------
    # Step 3: Determine scenario and execute
    # -------------------------------------------------------------------
    if len(inner_paths) > 0:
        # ---------------------------------------------------------------
        # Scenario B: Factory / Closure pattern
        # ---------------------------------------------------------------
        print("Detected Scenario B (Factory/Closure pattern).")

        # Phase 1: Reconstruct operator
        try:
            # Fix seed to match data generation
            np.random.seed(42)
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling load_and_preprocess_data (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute inner calls
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator (inner): {e}")
                traceback.print_exc()
                sys.exit(1)

            # Verification
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner call.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner call verification PASSED.")

    else:
        # ---------------------------------------------------------------
        # Scenario A: Simple function call
        # ---------------------------------------------------------------
        print("Detected Scenario A (Simple function).")

        expected = outer_data.get('output')

        try:
            # Fix seed to match data generation
            np.random.seed(42)
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling load_and_preprocess_data: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verification
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()