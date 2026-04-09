import sys
import os
import dill
import traceback
import numpy as np
import tempfile

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/sme_stellar_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
    # We need to override those paths with temp directories to avoid side effects
    # or permission issues. The args are (data_dict, result_dict, results_dir, assets_dir).
    # Let's create temp directories for results_dir and assets_dir.
    try:
        tmp_results_dir = tempfile.mkdtemp(prefix="test_results_")
        tmp_assets_dir = tempfile.mkdtemp(prefix="test_assets_")

        # The function signature is: evaluate_results(data_dict, result_dict, results_dir, assets_dir)
        # Replace results_dir and assets_dir in args with temp dirs
        modified_args = list(outer_args)
        if len(modified_args) >= 3:
            modified_args[2] = tmp_results_dir
        if len(modified_args) >= 4:
            modified_args[3] = tmp_assets_dir
        modified_args = tuple(modified_args)

        # Also handle if they were passed as kwargs
        modified_kwargs = dict(outer_kwargs)
        if 'results_dir' in modified_kwargs:
            modified_kwargs['results_dir'] = tmp_results_dir
        if 'assets_dir' in modified_kwargs:
            modified_kwargs['assets_dir'] = tmp_assets_dir

    except Exception as e:
        print(f"FAIL: Could not create temp directories: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            print("Phase 1: Reconstructing operator...")
            agent_operator = evaluate_results(*modified_args, **modified_kwargs)
            print(f"  Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Returned operator is not callable.")
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
                print("Phase 2: Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed: {inner_path}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        try:
            print("Phase 1: Running evaluate_results...")
            result = evaluate_results(*modified_args, **modified_kwargs)
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

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


if __name__ == "__main__":
    main()