import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def strip_rng_recursive(obj):
    """Remove 'rng' keys from dicts (and nested structures) since
    numpy Generator objects cannot be meaningfully compared."""
    if isinstance(obj, dict):
        return {k: strip_rng_recursive(v) for k, v in obj.items() if k != 'rng'}
    elif isinstance(obj, (list, tuple)):
        stripped = [strip_rng_recursive(item) for item in obj]
        return type(obj)(stripped)
    return obj


def main():
    data_paths = [
        '/data/yjh/cubical_cal_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file.")
        sys.exit(1)

    # Load outer data
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

    # Execute the function
    try:
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data with outer args.")
    except Exception as e:
        print(f"FAIL: Execution of load_and_preprocess_data failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
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

            if not callable(agent_result):
                print("FAIL: agent_result is not callable for Scenario B.")
                sys.exit(1)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Execution of inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = inner_data.get('output')
            # Strip rng from both
            result_clean = strip_rng_recursive(result)
            expected_clean = strip_rng_recursive(expected)

            try:
                passed, msg = recursive_check(expected_clean, result_clean)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {inner_path}")
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function")
        expected = outer_data.get('output')
        result = agent_result

        # Strip rng from both expected and result to avoid Generator comparison issues
        result_clean = strip_rng_recursive(result)
        expected_clean = strip_rng_recursive(expected)

        try:
            passed, msg = recursive_check(expected_clean, result_clean)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()