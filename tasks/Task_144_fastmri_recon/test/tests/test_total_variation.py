import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_total_variation import total_variation
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/fastmri_recon_sandbox_sandbox/run_code/std_data/standard_data_total_variation.pkl']

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
        print(f"Loaded outer data from {outer_path}")
        print(f"  func_name: {outer_data.get('func_name')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = total_variation(*outer_args, **outer_kwargs)
            print(f"Created operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data['output']
                print(f"Loaded inner data from {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        try:
            result = total_variation(*outer_args, **outer_kwargs)
            print(f"Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute function: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data['output']

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()