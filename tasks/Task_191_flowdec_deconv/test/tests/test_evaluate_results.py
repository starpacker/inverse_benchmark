import sys
import os
import dill
import traceback
import numpy as np

# Data paths provided
data_paths = ['/data/yjh/flowdec_deconv_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

# Separate outer vs inner paths
outer_path = None
inner_paths = []
for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    elif basename == 'standard_data_evaluate_results.pkl':
        outer_path = p

# Determine scenario
has_inner = len(inner_paths) > 0

def main():
    # ---- Phase 1: Load outer data and run evaluate_results ----
    if outer_path is None:
        print("FAIL: No outer data file found (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        from agent_evaluate_results import evaluate_results
    except Exception as e:
        print(f"FAIL: Could not import evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        agent_result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: evaluate_results raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Phase 2: Determine result and expected based on scenario ----
    if has_inner:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_result):
            print(f"FAIL: Expected evaluate_results to return a callable (operator), got {type(agent_result)}")
            sys.exit(1)

        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"FAIL: Could not load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)

        try:
            result = agent_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"FAIL: Operator execution raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function — result is the direct output
        result = agent_result
        expected = outer_output

    # ---- Phase 3: Comparison ----
    try:
        from verification_utils import recursive_check
    except Exception as e:
        print(f"FAIL: Could not import recursive_check: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"FAIL: recursive_check raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == '__main__':
    main()