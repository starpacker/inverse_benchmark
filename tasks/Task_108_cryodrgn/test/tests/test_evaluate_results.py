import sys
import os
import dill
import numpy as np
import traceback
import tempfile
import shutil

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/cryodrgn_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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
        print("ERROR: No outer data file found (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # The function writes files to results_dir and script_dir.
    # We need to redirect these to temporary directories so the test doesn't
    # fail due to permission issues or pollute the filesystem.
    # Identify which args are results_dir and script_dir (positional args 3 and 4,
    # or keyword args).
    # Convert args to list so we can modify
    outer_args = list(outer_args)

    # Create temporary directories for file outputs
    tmp_results_dir = tempfile.mkdtemp(prefix='test_eval_results_')
    tmp_script_dir = tempfile.mkdtemp(prefix='test_eval_script_')

    try:
        # The function signature is:
        # evaluate_results(ground_truth, reconstruction, projections, results_dir, script_dir,
        #                  n_projections, vol_size, noise_std)
        # Positional indices: 0=ground_truth, 1=reconstruction, 2=projections,
        #                     3=results_dir, 4=script_dir, 5=n_projections, 6=vol_size, 7=noise_std

        # Override results_dir and script_dir with temp dirs
        if 'results_dir' in outer_kwargs:
            outer_kwargs['results_dir'] = tmp_results_dir
        elif len(outer_args) > 3:
            outer_args[3] = tmp_results_dir

        if 'script_dir' in outer_kwargs:
            outer_kwargs['script_dir'] = tmp_script_dir
        elif len(outer_args) > 4:
            outer_args[4] = tmp_script_dir

        outer_args = tuple(outer_args)

        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("Scenario B detected: Factory/Closure pattern")
            try:
                agent_operator = evaluate_results(*outer_args, **outer_kwargs)
                print(f"Phase 1: Got operator of type {type(agent_operator)}")
            except Exception as e:
                print(f"ERROR in Phase 1 (creating operator): {e}")
                traceback.print_exc()
                sys.exit(1)

            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)

            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"Loaded inner data from: {inner_path}")
                except Exception as e:
                    print(f"ERROR loading inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)

                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                    print("Phase 2: Executed operator successfully")
                except Exception as e:
                    print(f"ERROR in Phase 2 (executing operator): {e}")
                    traceback.print_exc()
                    sys.exit(1)

                try:
                    passed, msg = recursive_check(inner_expected, result)
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print("TEST PASSED")
                except Exception as e:
                    print(f"ERROR during verification: {e}")
                    traceback.print_exc()
                    sys.exit(1)
        else:
            # Scenario A: Simple Function
            print("Scenario A detected: Simple function call")
            try:
                result = evaluate_results(*outer_args, **outer_kwargs)
                print(f"Function executed successfully, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR executing evaluate_results: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    finally:
        # Cleanup temporary directories
        try:
            shutil.rmtree(tmp_results_dir, ignore_errors=True)
            shutil.rmtree(tmp_script_dir, ignore_errors=True)
        except Exception:
            pass

    sys.exit(0)

if __name__ == '__main__':
    main()