import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/afm_force_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Override results_dir and assets_dir to use temp directories to avoid conflicts
    # The function writes files to these directories, so we use temporary ones
    import tempfile
    temp_results_dir = tempfile.mkdtemp(prefix='test_results_')
    temp_assets_dir = tempfile.mkdtemp(prefix='test_assets_')

    # The function signature is: evaluate_results(z, F_gt, F_recon, delta_f, results_dir, assets_dir)
    # args are positional: (z, F_gt, F_recon, delta_f, results_dir, assets_dir)
    # We need to replace results_dir and assets_dir with temp dirs
    outer_args_list = list(outer_args)
    if len(outer_args_list) >= 6:
        outer_args_list[4] = temp_results_dir
        outer_args_list[5] = temp_assets_dir
    else:
        # Try kwargs
        if 'results_dir' in outer_kwargs:
            outer_kwargs['results_dir'] = temp_results_dir
        if 'assets_dir' in outer_kwargs:
            outer_kwargs['assets_dir'] = temp_assets_dir
    outer_args = tuple(outer_args_list)

    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results executed (outer call).")
        except Exception as e:
            print(f"FAIL: Error executing evaluate_results (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Result of outer call is not callable. Falling back to Scenario A comparison.")
            # Fall back to Scenario A
            result = agent_operator
            expected = outer_output
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)

        # Process inner paths
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
            inner_output = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Inner operator executed.")
            except Exception as e:
                print(f"FAIL: Error executing inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_output, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple Function
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results executed (simple call).")
        except Exception as e:
            print(f"FAIL: Error executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()