import sys
import os
import dill
import traceback
import numpy as np

# Ensure the results directory exists for file I/O within the function
# We need to handle the case where the function writes files to results_dir

def main():
    try:
        from agent_evaluate_results import evaluate_results
        from verification_utils import recursive_check
    except ImportError as e:
        print(f"Import error: {e}")
        traceback.print_exc()
        sys.exit(1)

    data_paths = [
        '/data/yjh/poseidon_atm_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print(f"[INFO] Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Ensure results_dir exists so the function can write files
    try:
        if len(outer_args) > 0 and isinstance(outer_args[0], dict):
            results_dir = outer_args[0].get('results_dir', None)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
                print(f"[INFO] Created results_dir: {results_dir}")
        # Also check kwargs
        if 'data_dict' in outer_kwargs and isinstance(outer_kwargs['data_dict'], dict):
            results_dir = outer_kwargs['data_dict'].get('results_dir', None)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
                print(f"[INFO] Created results_dir: {results_dir}")
    except Exception as e:
        print(f"[WARN] Could not ensure results_dir exists: {e}")

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] evaluate_results returned: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR running evaluate_results (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing agent_operator (inner): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call")

        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"[INFO] evaluate_results returned type: {type(result)}")
        except Exception as e:
            print(f"ERROR running evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()