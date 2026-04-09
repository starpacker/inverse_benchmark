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
        '/data/yjh/straintool_geo_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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

    # Phase 2: Import and run function
    try:
        from agent_evaluate_results import evaluate_results
        print("Successfully imported evaluate_results")
    except Exception as e:
        print(f"FAIL: Could not import evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Outer call returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Outer call to evaluate_results failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Loaded inner data from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner call returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: Inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {ip}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner data verification passed for {os.path.basename(ip)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: evaluate_results call failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("PASS: Output verification passed")
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()