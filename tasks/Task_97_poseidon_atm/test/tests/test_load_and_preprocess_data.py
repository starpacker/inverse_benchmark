import sys
import os
import dill
import numpy as np
import traceback
import tempfile
import shutil

# Ensure the module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/poseidon_atm_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
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
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # The function writes to results_dir and may create files there.
    # We need to ensure the results_dir used during testing doesn't conflict.
    # We'll use a temporary directory for results_dir to ensure fresh generation.
    # But we need to match the original behavior. Let's inspect the args first.

    # The function signature:
    # load_and_preprocess_data(working_dir, results_dir, n_wave, wav_min, wav_max,
    #                           noise_level, seed, gt_params, r_star, m_planet,
    #                           n_layers, p_top, p_bottom)

    # We need to use a temp results_dir to avoid loading cached data that might not exist
    # or to ensure we generate fresh data for comparison.
    # Let's modify results_dir to a temp directory.

    try:
        # Create a temporary directory for results
        temp_dir = tempfile.mkdtemp(prefix='test_load_preprocess_')

        # Convert args to list so we can modify results_dir (index 1)
        modified_args = list(outer_args)
        if len(modified_args) > 1:
            original_results_dir = modified_args[1]
            modified_args[1] = os.path.join(temp_dir, 'results')
            # Also modify working_dir if needed
            if len(modified_args) > 0:
                modified_args[0] = temp_dir
        elif 'results_dir' in outer_kwargs:
            original_results_dir = outer_kwargs['results_dir']
            outer_kwargs = dict(outer_kwargs)
            outer_kwargs['results_dir'] = os.path.join(temp_dir, 'results')
            if 'working_dir' in outer_kwargs:
                outer_kwargs['working_dir'] = temp_dir

        modified_args = tuple(modified_args)

    except Exception as e:
        print(f"FAIL: Error setting up temp directory: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 1: Execute the function
    try:
        print("[INFO] Executing load_and_preprocess_data...")
        result = load_and_preprocess_data(*modified_args, **outer_kwargs)
        print(f"[INFO] Function returned type: {type(result)}")
    except Exception as e:
        print(f"FAIL: Error executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")
        if not callable(result):
            print(f"FAIL: Expected callable from outer function, got {type(result)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing inner operator: {e}")
                traceback.print_exc()
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner call: {msg}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner call verification passed.")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly
        print("[INFO] Scenario A detected: Simple function")

        # The expected output has 'results_dir' which will differ since we used temp dir.
        # We need to adjust the expected output or the result for comparison.
        # Let's fix the result's results_dir to match expected, since the key content
        # (wavelengths, spectrum_obs, spectrum_clean, sigma_obs) is what matters.

        if isinstance(result, dict) and isinstance(expected_output, dict):
            # Replace results_dir in result with the expected one for comparison
            if 'results_dir' in result and 'results_dir' in expected_output:
                result['results_dir'] = expected_output['results_dir']

        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)
            else:
                print("[INFO] Verification passed.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()