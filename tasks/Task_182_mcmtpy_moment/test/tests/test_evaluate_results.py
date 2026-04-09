import sys
import os
import dill
import numpy as np
import traceback

# Ensure the module directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/mcmtpy_moment_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Use a temporary output directory so we don't pollute the real one
    import tempfile
    temp_output_dir = tempfile.mkdtemp(prefix="test_evaluate_results_")

    # Override output_dir in kwargs or args to use temp directory
    # The function signature is: evaluate_results(data, inversion_result, output_dir='results')
    # Check if output_dir is in kwargs, otherwise patch it
    if 'output_dir' in outer_kwargs:
        outer_kwargs['output_dir'] = temp_output_dir
    elif len(outer_args) >= 3:
        outer_args = list(outer_args)
        outer_args[2] = temp_output_dir
        outer_args = tuple(outer_args)
    else:
        outer_kwargs['output_dir'] = temp_output_dir

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            print("[PHASE 1] Reconstructing operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to run evaluate_results (Phase 1): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: Expected a callable operator from evaluate_results, got non-callable.")
            sys.exit(1)

        # Sort inner paths for deterministic ordering
        inner_paths.sort()

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
                print("[PHASE 2] Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator (Phase 2): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"VERIFICATION FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"VERIFICATION PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        try:
            print("[PHASE 1] Running evaluate_results directly...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to run evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print("VERIFICATION PASSED")
        except Exception as e:
            print(f"ERROR: Verification exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()