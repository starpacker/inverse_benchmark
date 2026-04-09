import sys
import os
import dill
import traceback
import tempfile
import shutil

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/pyilc_cmb_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Separate outer vs inner paths
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

    # The function writes files to results_dir and assets_dir arguments.
    # We need to override those with temp directories to avoid side effects
    # and filesystem permission issues, while keeping the computation correct.
    # 
    # Looking at the function signature:
    #   evaluate_results(cmb_gt, cmb_rec, data, freqs_ghz, weights, results_dir, assets_dir)
    # results_dir is arg index 5, assets_dir is arg index 6

    # Create temporary directories for outputs
    tmp_results_dir = tempfile.mkdtemp(prefix='test_eval_results_')
    tmp_assets_dir = tempfile.mkdtemp(prefix='test_eval_assets_')

    try:
        # Replace results_dir and assets_dir in the args with temp dirs
        outer_args_list = list(outer_args)
        if len(outer_args_list) >= 7:
            outer_args_list[5] = tmp_results_dir
            outer_args_list[6] = tmp_assets_dir
        outer_args_modified = tuple(outer_args_list)

        # Also check kwargs
        modified_kwargs = dict(outer_kwargs)
        if 'results_dir' in modified_kwargs:
            modified_kwargs['results_dir'] = tmp_results_dir
        if 'assets_dir' in modified_kwargs:
            modified_kwargs['assets_dir'] = tmp_assets_dir

        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure pattern
            print("Scenario B detected: Factory/Closure pattern")

            # Phase 1: Create operator
            try:
                from agent_evaluate_results import evaluate_results
                agent_operator = evaluate_results(*outer_args_modified, **modified_kwargs)
                print("Phase 1: Operator created successfully.")
            except Exception as e:
                print(f"FAIL: Phase 1 - Could not create operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not callable(agent_operator):
                print("FAIL: Phase 1 - Result is not callable.")
                sys.exit(1)

            # Phase 2: Execute with inner data
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
                    print("Phase 2: Operator executed successfully.")
                except Exception as e:
                    print(f"FAIL: Phase 2 - Operator execution failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                # Compare
                try:
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"FAIL: Verification failed: {msg}")
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
            print("Scenario A detected: Simple function call")

            # Phase 1: Execute function
            try:
                from agent_evaluate_results import evaluate_results
                result = evaluate_results(*outer_args_modified, **modified_kwargs)
                print("Phase 1: Function executed successfully.")
            except Exception as e:
                print(f"FAIL: Phase 1 - Function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = outer_output

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    finally:
        # Clean up temp directories
        try:
            shutil.rmtree(tmp_results_dir, ignore_errors=True)
            shutil.rmtree(tmp_assets_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == '__main__':
    main()