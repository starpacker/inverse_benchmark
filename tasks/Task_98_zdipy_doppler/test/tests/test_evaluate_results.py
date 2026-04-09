import sys
import os
import dill
import traceback
import tempfile
import numpy as np

# Ensure we can import the target module and verification utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/zdipy_doppler_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # ── Step 1: Classify paths ──
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    # ── Step 2: Load outer data ──
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
    expected_output = outer_data.get('output', None)

    # ── Step 3: The function writes files to directories. 
    #    We need to override directory args with temp directories to avoid side effects ──
    # The function signature: evaluate_results(B_gt, B_rec, n_lat, n_lon, results_dir, assets_dir, working_dir)
    # args[4] = results_dir, args[5] = assets_dir, args[6] = working_dir
    # We replace these with temp dirs so we don't pollute or fail on missing dirs.

    try:
        # Convert args to a mutable list
        args_list = list(outer_args)

        # Create temp directories for results_dir, assets_dir, working_dir
        tmp_base = tempfile.mkdtemp(prefix="test_evaluate_results_")
        tmp_results = os.path.join(tmp_base, "results")
        tmp_assets = os.path.join(tmp_base, "assets")
        tmp_working = os.path.join(tmp_base, "working")

        # Replace directory arguments (indices 4, 5, 6) if they exist
        if len(args_list) > 4:
            args_list[4] = tmp_results
        if len(args_list) > 5:
            args_list[5] = tmp_assets
        if len(args_list) > 6:
            args_list[6] = tmp_working

        # Also check kwargs
        if 'results_dir' in outer_kwargs:
            outer_kwargs['results_dir'] = tmp_results
        if 'assets_dir' in outer_kwargs:
            outer_kwargs['assets_dir'] = tmp_assets
        if 'working_dir' in outer_kwargs:
            outer_kwargs['working_dir'] = tmp_working

        outer_args = tuple(args_list)
    except Exception as e:
        print(f"WARNING: Could not set up temp directories: {e}")
        traceback.print_exc()

    # ── Step 4: Determine scenario and execute ──
    if len(inner_paths) > 0:
        # ── Scenario B: Factory/Closure Pattern ──
        print("Detected Scenario B (Factory/Closure pattern)")

        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Phase 1: evaluate_results returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from Phase 1, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: operator returned: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, actual_result)
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
        # ── Scenario A: Simple Function ──
        print("Detected Scenario A (Simple function)")

        try:
            actual_result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"evaluate_results returned: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected_output, actual_result)
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


if __name__ == '__main__':
    main()