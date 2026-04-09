import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from agent_gs_refine_torch import gs_refine_torch
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/r2gaussian_ct_sandbox_sandbox/run_code/std_data/standard_data_gs_refine_torch.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_gs_refine_torch.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_gs_refine_torch.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
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

    # Scenario A: No inner paths, simple function call
    if len(inner_paths) == 0:
        print("Scenario A: Simple function call")

        # Fix seeds before running to match data generation
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            result = gs_refine_torch(*outer_args, **outer_kwargs)
            print("Function executed successfully.")
        except Exception as e:
            print(f"FAIL: Function execution raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Output mismatch.")
            print(f"  Details: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

    # Scenario B: Factory/Closure pattern
    else:
        print("Scenario B: Factory/Closure pattern")

        # Fix seeds before running to match data generation
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            agent_operator = gs_refine_torch(*outer_args, **outer_kwargs)
            print("Outer function executed successfully.")
        except Exception as e:
            print(f"FAIL: Outer function execution raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer function, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        for inner_path in inner_paths:
            print(f"Processing inner data: {inner_path}")

            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data. func_name: {inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("  Inner function executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner function execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = inner_data.get('output')

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Output mismatch for inner data {inner_path}")
                print(f"  Details: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()