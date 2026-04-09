import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/bilby_gw_sandbox_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_results(expected, actual):
    """
    Custom comparison that handles the non-deterministic parts of the output.
    We compare deterministic scalar values and array shapes/deterministic arrays,
    but skip objects that contain random noise state (ifos, h1, noisy_strain_fd, waveform_generator).
    """
    errors = []

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        passed, msg = recursive_check(expected, actual)
        return passed, msg

    # Check that all expected keys are present
    for key in expected:
        if key not in actual:
            errors.append(f"Missing key '{key}' in actual output.")
            continue

    # 1. Compare deterministic scalar values
    for key in ['true_chirp_mass', 'true_mass_ratio']:
        if key in expected:
            if key not in actual:
                errors.append(f"Missing key '{key}' in actual output.")
                continue
            exp_val = expected[key]
            act_val = actual[key]
            if not np.isclose(exp_val, act_val, rtol=1e-6, atol=1e-12):
                errors.append(f"Mismatch for '{key}': expected {exp_val}, got {act_val}")

    # 2. Compare freq_array (deterministic)
    if 'freq_array' in expected and 'freq_array' in actual:
        exp_arr = np.asarray(expected['freq_array'])
        act_arr = np.asarray(actual['freq_array'])
        if exp_arr.shape != act_arr.shape:
            errors.append(f"Shape mismatch for 'freq_array': expected {exp_arr.shape}, got {act_arr.shape}")
        elif not np.allclose(exp_arr, act_arr, rtol=1e-6, atol=1e-12):
            errors.append(f"Value mismatch for 'freq_array'")

    # 3. Compare gt_strain_fd (deterministic given same injection parameters)
    if 'gt_strain_fd' in expected and 'gt_strain_fd' in actual:
        exp_arr = np.asarray(expected['gt_strain_fd'])
        act_arr = np.asarray(actual['gt_strain_fd'])
        if exp_arr.shape != act_arr.shape:
            errors.append(f"Shape mismatch for 'gt_strain_fd': expected {exp_arr.shape}, got {act_arr.shape}")
        elif not np.allclose(exp_arr, act_arr, rtol=1e-4, atol=1e-30):
            max_diff = np.max(np.abs(exp_arr - act_arr))
            errors.append(f"Value mismatch for 'gt_strain_fd', max diff: {max_diff}")

    # 4. For noisy_strain_fd, just check shape (it's stochastic due to noise)
    if 'noisy_strain_fd' in expected and 'noisy_strain_fd' in actual:
        exp_arr = np.asarray(expected['noisy_strain_fd'])
        act_arr = np.asarray(actual['noisy_strain_fd'])
        if exp_arr.shape != act_arr.shape:
            errors.append(f"Shape mismatch for 'noisy_strain_fd': expected {exp_arr.shape}, got {act_arr.shape}")

    # 5. For ifos, check it's an InterferometerList with correct number of detectors and names
    if 'ifos' in expected and 'ifos' in actual:
        exp_ifos = expected['ifos']
        act_ifos = actual['ifos']
        if len(exp_ifos) != len(act_ifos):
            errors.append(f"ifos length mismatch: expected {len(exp_ifos)}, got {len(act_ifos)}")
        else:
            for i in range(len(exp_ifos)):
                if exp_ifos[i].name != act_ifos[i].name:
                    errors.append(f"ifos[{i}].name mismatch: expected {exp_ifos[i].name}, got {act_ifos[i].name}")
                if exp_ifos[i].minimum_frequency != act_ifos[i].minimum_frequency:
                    errors.append(f"ifos[{i}].minimum_frequency mismatch")
                if exp_ifos[i].maximum_frequency != act_ifos[i].maximum_frequency:
                    errors.append(f"ifos[{i}].maximum_frequency mismatch")

    # 6. For h1, check name and basic properties
    if 'h1' in expected and 'h1' in actual:
        if expected['h1'].name != actual['h1'].name:
            errors.append(f"h1.name mismatch: expected {expected['h1'].name}, got {actual['h1'].name}")

    # 7. For waveform_generator, check basic attributes
    if 'waveform_generator' in expected and 'waveform_generator' in actual:
        exp_wg = expected['waveform_generator']
        act_wg = actual['waveform_generator']
        if exp_wg.duration != act_wg.duration:
            errors.append(f"waveform_generator.duration mismatch: expected {exp_wg.duration}, got {act_wg.duration}")
        if exp_wg.sampling_frequency != act_wg.sampling_frequency:
            errors.append(f"waveform_generator.sampling_frequency mismatch")

    if errors:
        return False, "; ".join(errors)
    return True, "All checks passed."


def main():
    data_paths = ['/data/yjh/bilby_gw_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Execute the function
    try:
        print("[INFO] Executing load_and_preprocess_data with outer args/kwargs...")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if inner_paths:
        # Scenario B
        print("[INFO] Scenario B detected.")
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = compare_results(expected, actual_result)
            if not passed:
                print(f"FAIL: Test did not pass.\n  Message: {msg}")
                sys.exit(1)
    else:
        # Scenario A
        print("[INFO] Scenario A detected. Comparing direct output.")
        expected = outer_data.get('output')

        passed, msg = compare_results(expected, result)
        if not passed:
            print(f"FAIL: Test did not pass.\n  Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()