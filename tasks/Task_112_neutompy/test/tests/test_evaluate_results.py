import sys
import os
import dill
import traceback
import tempfile
import shutil

# Ensure the module can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/neutompy_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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
        print(f"  func_name: {outer_data.get('func_name')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    # The function writes files to results_dir and assets_dir.
    # We need to ensure those directories exist. The args contain directory paths.
    # We'll create temp directories to avoid conflicts, but we need to replace
    # the directory arguments with valid temp dirs so the function can write files.
    # Looking at the signature: evaluate_results(gt, recon, sinogram, results_dir, assets_dir)
    # args[3] = results_dir, args[4] = assets_dir

    # Create temporary directories for file outputs
    temp_results_dir = tempfile.mkdtemp(prefix='test_results_')
    temp_assets_dir = tempfile.mkdtemp(prefix='test_assets_')

    try:
        # Replace directory arguments with temp directories
        outer_args_list = list(outer_args)
        if len(outer_args_list) >= 5:
            outer_args_list[3] = temp_results_dir
            outer_args_list[4] = temp_assets_dir
        outer_args_modified = tuple(outer_args_list)

        # Also handle if dirs are in kwargs
        modified_kwargs = dict(outer_kwargs)
        if 'results_dir' in modified_kwargs:
            modified_kwargs['results_dir'] = temp_results_dir
        if 'assets_dir' in modified_kwargs:
            modified_kwargs['assets_dir'] = temp_assets_dir

        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure pattern
            print("Scenario B detected: Factory/Closure pattern")

            try:
                agent_operator = evaluate_results(*outer_args_modified, **modified_kwargs)
                print(f"Phase 1: evaluate_results returned: {type(agent_operator)}")
            except Exception as e:
                print(f"FAIL: Phase 1 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not callable(agent_operator):
                print("FAIL: Phase 1 result is not callable.")
                sys.exit(1)

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
                inner_expected = inner_data.get('output')

                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"FAIL: Phase 2 execution failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                try:
                    passed, msg = recursive_check(inner_expected, actual_result)
                except Exception as e:
                    print(f"FAIL: Verification error: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")

            print("TEST PASSED")
            sys.exit(0)

        else:
            # Scenario A: Simple function
            print("Scenario A detected: Simple function call")

            try:
                actual_result = evaluate_results(*outer_args_modified, **modified_kwargs)
                print(f"Function returned: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: Function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected_output, actual_result)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)

    finally:
        # Clean up temp directories
        try:
            shutil.rmtree(temp_results_dir, ignore_errors=True)
            shutil.rmtree(temp_assets_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == '__main__':
    main()