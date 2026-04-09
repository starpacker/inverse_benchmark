import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/diffpy_pdf_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for forward_operator.")
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
        print(f"  num args: {len(outer_args)}, num kwargs: {len(outer_kwargs)}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute function
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print(f"Created operator/closure from forward_operator. Type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Result of forward_operator is not callable. Got type: {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"\nLoaded inner data [{idx}] from: {inner_path}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
                print(f"  num args: {len(inner_args)}, num kwargs: {len(inner_kwargs)}")
            except Exception as e:
                print(f"FAIL: Could not load inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator with inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL [inner {idx}]: {msg}")
                    all_passed = False
                else:
                    print(f"PASSED [inner {idx}]")
            except Exception as e:
                print(f"FAIL: Verification error [inner {idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            print("\nTEST FAILED")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print(f"Executed forward_operator. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                print("\nTEST FAILED")
                sys.exit(1)
            else:
                print("\nTEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()