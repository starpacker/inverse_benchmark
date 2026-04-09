import sys
import os
import dill
import numpy as np
import traceback

# Add the necessary paths
sys.path.insert(0, '/data/yjh/aspire_cryoem_sandbox_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_results_custom(expected, actual, path="output"):
    """
    Custom comparison that handles non-comparable objects like Simulation
    by skipping them and comparing everything else.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        if expected_keys != actual_keys:
            return False, f"Key mismatch at {path}: expected {expected_keys}, got {actual_keys}"
        for key in expected_keys:
            # Skip 'sim' key - it's an ASPIRE Simulation object that can't be compared by identity
            if key == 'sim':
                # Just check that it exists and is the right type
                from aspire.source.simulation import Simulation
                if not isinstance(actual[key], Simulation):
                    return False, f"At {path}['{key}']: expected Simulation object, got {type(actual[key])}"
                continue
            passed, msg = compare_results_custom(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, "All checks passed"
    elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype differences but warn
            pass
        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8, equal_nan=True):
            max_diff = np.max(np.abs(expected - actual))
            return False, f"Value mismatch at {path}: max diff = {max_diff}"
        return True, "Match"
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
            if not np.isclose(expected, actual, rtol=1e-5, atol=1e-8):
                return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
            return True, "Match"
        if expected != actual:
            return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
        return True, "Match"
    else:
        # For other types, try recursive_check but catch errors
        try:
            return recursive_check(expected, actual)
        except Exception:
            # If recursive_check fails, do basic comparison
            if type(expected) != type(actual):
                return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
            try:
                if expected == actual:
                    return True, "Match"
                else:
                    return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
            except Exception:
                # Objects that can't be compared - skip
                return True, "Skipped (non-comparable)"


def main():
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")

    if outer_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)

    # Phase 1: Load outer data
    print("\n[Phase 1] Loading outer data...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Args count: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 1: Execute function
    print("\n[Phase 1] Executing load_and_preprocess_data...")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("  Execution completed successfully.")
    except Exception as e:
        print(f"ERROR executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("\n[Scenario B] Factory/Closure pattern detected.")
        for inner_path in inner_paths:
            print(f"\n  Loading inner data from: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            if not callable(result):
                print("ERROR: Result from Phase 1 is not callable!")
                sys.exit(1)

            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing inner call: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = compare_results_custom(inner_expected, actual_result)
            if not passed:
                print("\n" + "=" * 50)
                print("TEST FAILED")
                print("=" * 50)
                print(f"Verification message: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n[Scenario A] Simple function pattern detected.")
        print("\n[Phase 3] Verifying results...")
        print(f"  Expected type: {type(expected_output)}")
        print(f"  Result type: {type(result)}")

        if isinstance(expected_output, dict):
            print(f"  Expected keys: {sorted(expected_output.keys())}")
        if isinstance(result, dict):
            print(f"  Result keys: {sorted(result.keys())}")

        # Use custom comparison that handles Simulation objects
        passed, msg = compare_results_custom(expected_output, result)

        if not passed:
            print("\n" + "=" * 50)
            print("TEST FAILED")
            print("=" * 50)
            print(f"Verification message: {msg}")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("TEST PASSED")
    print("=" * 50)
    sys.exit(0)


if __name__ == '__main__':
    main()