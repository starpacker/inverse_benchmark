import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/fastmri_recon_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("FAIL: No outer data file found for evaluate_results.")
        sys.exit(1)

    # --- Phase 1: Load outer data and execute function ---
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

    # For evaluate_results, if save_dir is provided, override it to a temp dir
    # to avoid polluting real directories during testing
    # We need to handle this carefully: the function writes files if save_dir is not None
    # We'll let it run as-is since we just need to compare the returned metrics dict

    try:
        # If save_dir was provided in the original call, redirect to a temp location
        # to avoid side effects but still exercise the same code path
        import tempfile
        temp_save_dir = None

        # Check if save_dir is in kwargs
        if 'save_dir' in outer_kwargs and outer_kwargs['save_dir'] is not None:
            temp_save_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
            modified_kwargs = dict(outer_kwargs)
            modified_kwargs['save_dir'] = temp_save_dir
            actual_result = evaluate_results(*outer_args, **modified_kwargs)
        else:
            # Check positional args - save_dir is the 5th argument (index 4)
            if len(outer_args) > 4 and outer_args[4] is not None:
                temp_save_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
                modified_args = list(outer_args)
                modified_args[4] = temp_save_dir
                actual_result = evaluate_results(*modified_args, **outer_kwargs)
            else:
                actual_result = evaluate_results(*outer_args, **outer_kwargs)

        print(f"  evaluate_results executed successfully.")
        print(f"  Result type: {type(actual_result)}")
    except Exception as e:
        print(f"FAIL: evaluate_results raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern")

        if not callable(actual_result):
            print(f"FAIL: Expected callable from evaluate_results, got {type(actual_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from: {inner_path}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = actual_result(*inner_args, **inner_kwargs)
                print(f"  Inner call executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner call raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner call.")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner call verification PASSED.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function call")

        expected = outer_data.get('output')
        result = actual_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Verification PASSED.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()