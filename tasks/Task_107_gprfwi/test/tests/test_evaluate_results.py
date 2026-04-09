import sys
import os
import dill
import numpy as np
import traceback
import tempfile
import shutil

# Ensure the module can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/gprfwi_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # The function writes files to results_dir and assets_dir (args[3] and args[4]).
    # We need to use temporary directories to avoid side effects but still let the function run.
    # Replace the directory arguments with temp directories.
    temp_results_dir = None
    temp_assets_dir = None
    try:
        temp_results_dir = tempfile.mkdtemp(prefix="test_results_")
        temp_assets_dir = tempfile.mkdtemp(prefix="test_assets_")

        # Build modified args: replace results_dir and assets_dir with temp dirs
        modified_args = list(outer_args)
        if len(modified_args) >= 5:
            modified_args[3] = temp_results_dir
            modified_args[4] = temp_assets_dir
        modified_args = tuple(modified_args)

        modified_kwargs = dict(outer_kwargs)
        if 'results_dir' in modified_kwargs:
            modified_kwargs['results_dir'] = temp_results_dir
        if 'assets_dir' in modified_kwargs:
            modified_kwargs['assets_dir'] = temp_assets_dir
    except Exception as e:
        print(f"FAIL: Could not set up temp directories: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")
        try:
            agent_operator = evaluate_results(*modified_args, **modified_kwargs)
            print(f"Phase 1 complete. agent_operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2 execution complete. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A (Simple function)")
        try:
            result = evaluate_results(*modified_args, **modified_kwargs)
            print(f"Execution complete. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
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
                print("Verification passed.")
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp directories
    try:
        if temp_results_dir and os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)
        if temp_assets_dir and os.path.exists(temp_assets_dir):
            shutil.rmtree(temp_assets_dir)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()