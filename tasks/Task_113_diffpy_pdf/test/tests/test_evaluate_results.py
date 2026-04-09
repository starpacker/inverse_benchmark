import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/diffpy_pdf_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("FAIL: No outer data file found for evaluate_results.")
        sys.exit(1)

    # ------- Phase 1: Load outer data -------
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
    expected_output = outer_data.get('output', None)

    # ------- Determine scenario -------
    has_inner = len(inner_paths) > 0

    if has_inner:
        # ===== Scenario B: Factory/Closure Pattern =====
        try:
            from agent_evaluate_results import evaluate_results
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: operator/closure created successfully.")
        except Exception as e:
            print(f"FAIL: Could not create operator from evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Could not execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"PASS: Verification succeeded for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification exception for {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        # ===== Scenario A: Simple Function =====
        try:
            from agent_evaluate_results import evaluate_results
            actual_result = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: evaluate_results executed successfully.")
        except Exception as e:
            print(f"FAIL: Could not execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()