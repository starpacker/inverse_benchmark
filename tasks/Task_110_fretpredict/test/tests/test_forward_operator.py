import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/fretpredict_sandbox_sandbox/run_code')

from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    data_paths = ['/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

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
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Load outer data
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

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        try:
            # Reset seeds to match the capture environment
            np.random.seed(42)
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: forward_operator did not return a callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {inner_path}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")
        expected = outer_data.get('output')

        # The function uses np.random.randn for noise. We need to reproduce
        # the exact same random state. In gen_data_code, _fix_seeds_(42) is
        # called at module load time before forward_operator is invoked.
        # We must replicate that exact seed state.
        import random
        np.random.seed(42)
        random.seed(42)
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        except ImportError:
            pass

        try:
            result = forward_operator(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_operator raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()