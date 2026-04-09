import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/arim_ndt_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Separate outer (direct function) and inner (parent/closure) paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # If results_dir is in the args, ensure the directory can be created
    # The function creates the directory and saves files into it, so we use a temp dir
    # to avoid conflicts. We need to find where results_dir is in the arguments.
    # Based on the signature, results_dir is the last positional argument (index 16).
    # We'll use a temporary directory to avoid polluting the filesystem.
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="test_load_preprocess_")

    # Modify results_dir in outer_args if present
    # The function signature has results_dir as the 17th argument (index 16)
    if len(outer_args) > 16:
        outer_args = list(outer_args)
        outer_args[16] = temp_dir
        outer_args = tuple(outer_args)
    elif 'results_dir' in outer_kwargs:
        outer_kwargs['results_dir'] = temp_dir

    # Also update expected output's results_dir to match if it's a dict
    if isinstance(outer_output, dict) and 'results_dir' in outer_output:
        outer_output = dict(outer_output)
        outer_output['results_dir'] = temp_dir

    # ---- Determine Scenario ----
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            print("Executing load_and_preprocess_data to get operator...")
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"  Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error executing load_and_preprocess_data (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing operator (inner): {e}")
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
                print(f"FAIL: Error during verification (inner): {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            print("Executing load_and_preprocess_data...")
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error executing load_and_preprocess_data: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("  Verification PASSED.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()