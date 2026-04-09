import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/phasorpy_flim_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
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
        print("FAIL: No outer data file found (standard_data_forward_operator.pkl)")
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
        print(f"  args count: {len(outer_args)}")
        print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute the function
    try:
        agent_result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully.")
        print(f"  Result type: {type(agent_result)}")
        if hasattr(agent_result, 'shape'):
            print(f"  Result shape: {agent_result.shape}")
    except Exception as e:
        print(f"FAIL: forward_operator execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 3: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify that agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from forward_operator, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                print(f"\nLoaded inner data from: {os.path.basename(inner_path)}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
                print(f"  args count: {len(inner_args)}")
                print(f"  kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner execution successful. Result type: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"  Verification PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function, comparing output directly.")

        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
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