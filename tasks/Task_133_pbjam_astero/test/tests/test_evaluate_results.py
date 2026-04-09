import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pbjam_astero_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Created operator successfully.")
        except Exception as e:
            print(f"ERROR in Phase 1 (creating operator): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable.")
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {ip}")
            except Exception as e:
                print(f"ERROR loading inner data from {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Executed operator successfully.")
            except Exception as e:
                print(f"ERROR in Phase 2 (executing operator): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
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
        # Scenario A: Simple function call
        expected = outer_data.get('output')

        # Redirect results_dir to a temporary directory to avoid conflicts
        # The function writes files, so we use a temp dir
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')

        # Replace results_dir argument if present
        # Check if results_dir is in args or kwargs
        modified_args = list(outer_args)
        modified_kwargs = dict(outer_kwargs)

        # The function signature: evaluate_results(gt_freq_array, recon_result, clean_spectrum, metadata, results_dir)
        # results_dir is the 5th positional arg (index 4)
        if 'results_dir' in modified_kwargs:
            modified_kwargs['results_dir'] = temp_dir
        elif len(modified_args) >= 5:
            modified_args[4] = temp_dir
        else:
            modified_kwargs['results_dir'] = temp_dir

        try:
            result = evaluate_results(*modified_args, **modified_kwargs)
            print("Phase 1: Executed function successfully.")
        except Exception as e:
            print(f"ERROR executing function: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()