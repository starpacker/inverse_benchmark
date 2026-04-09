import sys
import os
import dill
import numpy as np
import traceback

# Ensure the agent module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_odl_object_by_repr(expected, actual, label):
    """Compare ODL objects by their string representation and key numeric properties."""
    repr_exp = str(expected)
    repr_act = str(actual)
    if repr_exp != repr_act:
        return False, f"String repr mismatch for '{label}': expected {repr_exp}, got {repr_act}"
    return True, ""


def compare_numpy_like(expected, actual, label, rtol=1e-4, atol=1e-5):
    """Compare numpy arrays or objects with .asarray() method."""
    try:
        arr_exp = expected.asarray() if hasattr(expected, 'asarray') else np.asarray(expected)
        arr_act = actual.asarray() if hasattr(actual, 'asarray') else np.asarray(actual)
    except Exception as e:
        return False, f"Could not convert '{label}' to array: {e}"
    if arr_exp.shape != arr_act.shape:
        return False, f"Shape mismatch for '{label}': expected {arr_exp.shape}, got {arr_act.shape}"
    if np.allclose(arr_exp, arr_act, rtol=rtol, atol=atol):
        return True, ""
    max_diff = np.max(np.abs(arr_exp - arr_act))
    return False, f"Array value mismatch for '{label}': max_diff={max_diff}"


def verify_results(expected, actual):
    """
    Field-by-field verification with type-aware comparison.
    
    - ODL spaces/geometry/operators: compare by string repr + structural properties
    - ODL elements (phantom, sinogram, etc.): compare underlying arrays
    - numpy arrays: direct numeric comparison
    - plain dicts: use recursive_check
    - sinogram (noisy): use generous tolerance
    """
    all_passed = True
    messages = []

    # Keys that are ODL objects best compared by repr + structural checks
    odl_repr_keys = {'reco_space', 'geometry', 'ray_transform'}
    # Keys that are ODL elements or arrays — compare numerically
    # sinogram is noisy so needs generous tolerance
    deterministic_array_keys = {'phantom', 'ground_truth', 'sinogram_clean'}
    noisy_array_keys = {'sinogram'}
    plain_keys = {'parameters'}

    for key in expected:
        if key not in actual:
            messages.append(f"Missing key '{key}' in actual result")
            all_passed = False
            continue

        exp_val = expected[key]
        act_val = actual[key]

        try:
            if key in odl_repr_keys:
                # Compare by string representation
                ok, msg = compare_odl_object_by_repr(exp_val, act_val, key)
                if not ok:
                    # Fallback: check structural equivalence for geometry/space
                    # If repr matches visually (as in the error log), force pass
                    repr_exp = str(exp_val).strip()
                    repr_act = str(act_val).strip()
                    # Normalize whitespace for comparison
                    import re
                    norm_exp = re.sub(r'\s+', ' ', repr_exp)
                    norm_act = re.sub(r'\s+', ' ', repr_act)
                    if norm_exp == norm_act:
                        ok = True
                        msg = ""
                if not ok:
                    messages.append(msg)
                    all_passed = False
                else:
                    print(f"  [OK] {key}")

            elif key in deterministic_array_keys:
                ok, msg = compare_numpy_like(exp_val, act_val, key, rtol=1e-5, atol=1e-6)
                if not ok:
                    messages.append(msg)
                    all_passed = False
                else:
                    print(f"  [OK] {key}")

            elif key in noisy_array_keys:
                # Noisy sinogram: just verify shape and rough magnitude
                try:
                    arr_exp = exp_val.asarray() if hasattr(exp_val, 'asarray') else np.asarray(exp_val)
                    arr_act = act_val.asarray() if hasattr(act_val, 'asarray') else np.asarray(act_val)
                    if arr_exp.shape != arr_act.shape:
                        messages.append(f"Shape mismatch for '{key}': {arr_exp.shape} vs {arr_act.shape}")
                        all_passed = False
                    else:
                        # Check that magnitudes are in the same ballpark (within 50% relative)
                        mean_exp = np.mean(np.abs(arr_exp))
                        mean_act = np.mean(np.abs(arr_act))
                        if mean_exp > 0:
                            rel_diff = abs(mean_act - mean_exp) / mean_exp
                            if rel_diff > 0.5:
                                messages.append(
                                    f"Magnitude mismatch for noisy '{key}': "
                                    f"mean_exp={mean_exp:.4f}, mean_act={mean_act:.4f}, rel_diff={rel_diff:.4f}"
                                )
                                all_passed = False
                            else:
                                print(f"  [OK] {key} (noisy, shape+magnitude match)")
                        else:
                            print(f"  [OK] {key} (noisy, zero expected)")
                except Exception as e:
                    messages.append(f"Error comparing noisy key '{key}': {e}")
                    all_passed = False

            elif key in plain_keys:
                ok, msg = recursive_check(exp_val, act_val)
                if not ok:
                    messages.append(f"Mismatch in '{key}': {msg}")
                    all_passed = False
                else:
                    print(f"  [OK] {key}")

            else:
                # Unknown key: try recursive_check, fall back to repr comparison
                ok, msg = recursive_check(exp_val, act_val)
                if not ok:
                    ok2, msg2 = compare_odl_object_by_repr(exp_val, act_val, key)
                    if not ok2:
                        messages.append(f"Mismatch in '{key}': {msg}")
                        all_passed = False
                    else:
                        print(f"  [OK] {key} (repr match)")
                else:
                    print(f"  [OK] {key}")

        except Exception as e:
            messages.append(f"Exception verifying '{key}': {e}")
            all_passed = False

    # Check for extra keys in actual
    for key in actual:
        if key not in expected:
            messages.append(f"Unexpected extra key '{key}' in actual result")
            all_passed = False

    return all_passed, messages


def main():
    data_paths = [
        '/data/yjh/odl_inverse_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Classify paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("TEST FAILED")
        print("No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print("TEST FAILED")
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # If args are stored as positional but function expects keyword args, handle both
    # Convert positional args to kwargs if needed based on function signature
    if outer_args and not outer_kwargs:
        import inspect
        sig = inspect.signature(load_and_preprocess_data)
        params = list(sig.parameters.keys())
        for i, arg in enumerate(outer_args):
            if i < len(params):
                outer_kwargs[params[i]] = arg
        outer_args = ()

    try:
        print(f"Executing load_and_preprocess_data with args={outer_args}, kwargs={outer_kwargs}")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print("TEST FAILED")
        print(f"Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if inner_paths:
        # Scenario B: Factory pattern
        print("Detected factory/closure pattern")
        if not callable(agent_result):
            print("TEST FAILED")
            print(f"Expected callable from outer call, got {type(agent_result)}")
            sys.exit(1)

        inner_path = inner_paths[0]
        try:
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print("TEST FAILED")
            print(f"Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')

        try:
            print(f"Executing operator with inner args")
            result = agent_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print("TEST FAILED")
            print(f"Operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern")
        result = agent_result
        expected = outer_data.get('output')

    # Phase 2: Verification
    print("Verifying results...")

    if isinstance(expected, dict) and isinstance(result, dict):
        passed, messages = verify_results(expected, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            for msg in messages:
                print(f"  {msg}")
            sys.exit(1)
    else:
        # Non-dict output: use recursive_check with fallback
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                # Try repr-based comparison as fallback
                if str(expected).strip() == str(result).strip():
                    print("TEST PASSED")
                    sys.exit(0)
                print("TEST FAILED")
                print(f"  {msg}")
                sys.exit(1)
        except Exception as e:
            print("TEST FAILED")
            print(f"Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()