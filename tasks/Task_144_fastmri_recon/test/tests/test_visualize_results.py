import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_visualize_results import visualize_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/fastmri_recon_sandbox_sandbox/run_code/std_data/standard_data_visualize_results.pkl'
    ]

    # Separate outer vs inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for visualize_results.")
        sys.exit(1)

    # -----------------------------------------------------------
    # Phase 1: Load outer data and run the function
    # -----------------------------------------------------------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    # If save_path arg points to a specific location, we may want to redirect
    # to a temp path to avoid overwriting important files. Inspect args:
    # visualize_results(gt, zero_filled, cs_recon, metrics_zf, metrics_cs, save_path)
    # save_path is the 6th positional arg (index 5).
    # We'll use a temp path to avoid side-effects, but keep the same extension.
    import tempfile
    temp_save_path = None
    modified_args = list(outer_args)
    modified_kwargs = dict(outer_kwargs)

    # Check if save_path is in args or kwargs
    if len(modified_args) >= 6:
        original_save_path = modified_args[5]
        temp_dir = tempfile.mkdtemp()
        temp_save_path = os.path.join(temp_dir, os.path.basename(str(original_save_path)))
        modified_args[5] = temp_save_path
    elif 'save_path' in modified_kwargs:
        original_save_path = modified_kwargs['save_path']
        temp_dir = tempfile.mkdtemp()
        temp_save_path = os.path.join(temp_dir, os.path.basename(str(original_save_path)))
        modified_kwargs['save_path'] = temp_save_path

    try:
        result = visualize_results(*modified_args, **modified_kwargs)
        print("Phase 1: visualize_results executed successfully.")
    except Exception as e:
        print(f"FAIL: visualize_results raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------
    # Phase 2: Determine scenario and verify
    # -----------------------------------------------------------
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s).")

        if not callable(result):
            print(f"FAIL: Expected callable from visualize_results, got {type(result)}")
            sys.exit(1)

        agent_operator = result
        all_passed = True

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')

            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Inner execution succeeded for {os.path.basename(ip)}")
            except Exception as e:
                print(f"FAIL: agent_operator raised exception for {os.path.basename(ip)}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(ip)}: {msg}")
                    all_passed = False
                else:
                    print(f"  Verification passed for {os.path.basename(ip)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised exception for {os.path.basename(ip)}: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function — result from Phase 1 IS the result
        print("Scenario A detected: Simple function call.")

        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()