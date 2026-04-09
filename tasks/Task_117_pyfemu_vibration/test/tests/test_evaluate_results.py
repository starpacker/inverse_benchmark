import sys
import os
import dill
import numpy as np
import traceback
import tempfile
import shutil

# Ensure the module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/pyfemu_vibration_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

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
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # The function writes files to results_dir and assets_dir.
    # We need to redirect those to temp directories to avoid polluting the filesystem
    # and to avoid permission issues. We patch the args accordingly.
    # Based on the function signature:
    # evaluate_results(d_gt, d_recon, freqs_gt, freqs_recon, modes_gt, modes_recon,
    #                  freqs_obs, n_modes, n_elem, L_total, results_dir, assets_dir)
    # results_dir is arg index 10, assets_dir is arg index 11

    temp_results_dir = tempfile.mkdtemp(prefix="test_eval_results_")
    temp_assets_dir = tempfile.mkdtemp(prefix="test_eval_assets_")

    try:
        # Convert args to list so we can modify directory paths
        args_list = list(outer_args)

        # Replace results_dir and assets_dir with temp directories
        # The function has 12 positional args; indices 10 and 11 are the dirs
        if len(args_list) >= 12:
            args_list[10] = temp_results_dir
            args_list[11] = temp_assets_dir
        else:
            # Check if they're in kwargs
            if 'results_dir' in outer_kwargs:
                outer_kwargs['results_dir'] = temp_results_dir
            if 'assets_dir' in outer_kwargs:
                outer_kwargs['assets_dir'] = temp_assets_dir

        modified_args = tuple(args_list)

        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure pattern
            print("Scenario B detected: Factory/Closure pattern")

            try:
                agent_operator = evaluate_results(*modified_args, **outer_kwargs)
                print(f"Phase 1: evaluate_results returned: {type(agent_operator)}")
            except Exception as e:
                print(f"FAIL: Phase 1 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not callable(agent_operator):
                print("FAIL: Phase 1 result is not callable.")
                sys.exit(1)

            # Sort inner paths for deterministic order
            inner_paths.sort()

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
                expected = inner_data.get('output', None)

                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"FAIL: Phase 2 execution failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                try:
                    passed, msg = recursive_check(expected, result)
                except Exception as e:
                    print(f"FAIL: Verification error: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {os.path.basename(inner_path)}")

        else:
            # Scenario A: Simple function
            print("Scenario A detected: Simple function call")

            try:
                result = evaluate_results(*modified_args, **outer_kwargs)
                print(f"Function returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: Function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = outer_output

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    finally:
        # Clean up temp directories
        try:
            shutil.rmtree(temp_results_dir, ignore_errors=True)
            shutil.rmtree(temp_assets_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()