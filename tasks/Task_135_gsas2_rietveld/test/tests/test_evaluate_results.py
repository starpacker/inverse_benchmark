import sys
import os
import dill
import traceback
import numpy as np

# Ensure the module directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/gsas2_rietveld_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # ── Phase 1: Load outer data ──
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # ── Determine scenario ──
    if len(inner_paths) > 0:
        # Scenario B: Factory / Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        # Run outer function to get the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from outer call.")
        except Exception as e:
            print(f"FAIL: Error running evaluate_results (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        all_passed = True
        for ip in sorted(inner_paths):
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL (inner {os.path.basename(ip)}): {msg}")
                all_passed = False
            else:
                print(f"PASS (inner {os.path.basename(ip)})")

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        # Modify results_dir to use a temp directory to avoid conflicts
        # The third argument is results_dir; we use a temporary one
        modified_args = list(outer_args)
        if len(modified_args) >= 3:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
            modified_args[2] = tmp_dir
            print(f"Using temporary results_dir: {tmp_dir}")

        modified_kwargs = dict(outer_kwargs)
        if 'results_dir' in modified_kwargs:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
            modified_kwargs['results_dir'] = tmp_dir
            print(f"Using temporary results_dir: {tmp_dir}")

        try:
            result = evaluate_results(*modified_args, **modified_kwargs)
            print("Successfully ran evaluate_results.")
        except Exception as e:
            print(f"FAIL: Error running evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
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