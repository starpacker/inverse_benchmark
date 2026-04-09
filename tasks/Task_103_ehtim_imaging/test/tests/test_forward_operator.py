import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/ehtim_imaging_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

    # Step 1: Classify data files into outer (direct) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (standard_data_forward_operator.pkl) found.")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Step 3: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            print(f"  forward_operator returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception during Phase 1 (operator creation).")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from forward_operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  agent_operator returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception during Phase 2 (execution).")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception.")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Output mismatch for inner data: {inner_path}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  PASSED for inner data: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Phase 1: Execute function directly
        try:
            result = forward_operator(*outer_args, **outer_kwargs)
            print(f"  forward_operator returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Output mismatch.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("  PASSED for outer data.")

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()