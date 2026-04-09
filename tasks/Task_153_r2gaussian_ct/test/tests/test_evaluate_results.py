import sys
import os
import dill
import torch
import numpy as np
import traceback
import tempfile

# Ensure the module directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/r2gaussian_ct_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Step 1: Separate outer and inner paths
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

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Step 3: Override results_dir to a temp directory to avoid polluting the original results
    # The function writes files to disk; we use a temp dir to keep things clean.
    tmp_results_dir = tempfile.mkdtemp(prefix="test_evaluate_results_")

    # Inject results_dir into kwargs if not already provided, or override
    # Check if results_dir is in args (3rd positional arg) or kwargs
    modified_args = list(outer_args)
    modified_kwargs = dict(outer_kwargs)

    # The function signature: evaluate_results(data, result, results_dir=None)
    # If len(args) >= 3, the 3rd arg is results_dir
    if len(modified_args) >= 3:
        modified_args[2] = tmp_results_dir
    else:
        modified_kwargs['results_dir'] = tmp_results_dir

    # Step 4: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B: Factory/Closure pattern detected")
        try:
            agent_operator = evaluate_results(*modified_args, **modified_kwargs)
            print(f"[INFO] Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Agent operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"TEST PASSED (inner: {os.path.basename(inner_path)})")
    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A: Simple function pattern detected")
        try:
            result = evaluate_results(*modified_args, **modified_kwargs)
            print(f"[INFO] Function returned: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(tmp_results_dir, ignore_errors=True)
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()