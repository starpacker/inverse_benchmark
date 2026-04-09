import sys
import os
import dill
import numpy as np
import traceback

# Ensure matplotlib uses non-interactive backend before any other imports
import matplotlib
matplotlib.use('Agg')

from agent__visualize_results import _visualize_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pyxrf_fluor_sandbox_sandbox/run_code/std_data/standard_data__visualize_results.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
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
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Check if save_path argument exists and modify it to a temp path to avoid overwriting
    # The function saves a file, so we need to handle the save_path argument
    # Looking at the signature: energy, noisy_spectrum, clean_spectrum, recon_result,
    #   gt_concentrations, metrics, save_path, e_min, e_max, detector_fwhm, xrf_lines
    # save_path is at index 6 in positional args
    if len(outer_args) > 6:
        original_save_path = outer_args[6]
        # Use a temporary path for the test
        test_save_path = '/tmp/test_visualize_results_output.png'
        outer_args = list(outer_args)
        outer_args[6] = test_save_path
        outer_args = tuple(outer_args)
    elif 'save_path' in outer_kwargs:
        original_save_path = outer_kwargs['save_path']
        test_save_path = '/tmp/test_visualize_results_output.png'
        outer_kwargs = dict(outer_kwargs)
        outer_kwargs['save_path'] = test_save_path
    else:
        test_save_path = None

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B: Factory/Closure pattern detected.")
        try:
            agent_operator = _visualize_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Agent operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A: Simple function pattern detected.")
        try:
            result = _visualize_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Function executed. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute function: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")

                # Additional verification: check that the file was actually saved
                if test_save_path and os.path.exists(test_save_path):
                    file_size = os.path.getsize(test_save_path)
                    print(f"[INFO] Output file created at {test_save_path}, size={file_size} bytes")
                    # Clean up
                    try:
                        os.remove(test_save_path)
                    except:
                        pass

                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()