import sys
import os
import dill
import traceback
import numpy as np

# Ensure the proper module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/tttrlib_flim_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            print("Phase 1: Constructing operator from evaluate_results...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"  Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR in Phase 1 (constructing operator): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: The returned operator is not callable. Falling back to Scenario A comparison.")
            # Fall back: compare outer result directly
            result = agent_operator
            expected = outer_output
        else:
            # Phase 2: Load inner data and execute
            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"Loaded inner data from: {inner_path}")
                except Exception as e:
                    print(f"ERROR loading inner data from {inner_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)

                try:
                    print("Phase 2: Executing operator with inner args...")
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR in Phase 2 (executing operator): {e}")
                    traceback.print_exc()
                    sys.exit(1)

                # Comparison
                try:
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED for inner path {inner_path}: {msg}")
                        sys.exit(1)
                    else:
                        print(f"CHECK PASSED for inner path: {inner_path}")
                except Exception as e:
                    print(f"ERROR during comparison: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple function call
        try:
            print("Scenario A: Running evaluate_results directly...")
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR running evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

    # Final comparison (for Scenario A or fallback)
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


if __name__ == '__main__':
    main()