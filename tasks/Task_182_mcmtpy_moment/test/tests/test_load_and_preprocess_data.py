import sys
import os
import dill
import numpy as np
import traceback

# Fix random seeds to match the data generation environment
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except ImportError:
    torch = None

import random
random.seed(42)

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/mcmtpy_moment_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        # Phase 1: Create operator
        try:
            # Reset seeds before calling, to match generation environment
            np.random.seed(42)
            random.seed(42)
            if torch is not None:
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"[INFO] Agent operator created, type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Agent operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner test passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: Simple function")

        try:
            # Reset seeds before calling, to match generation environment
            np.random.seed(42)
            random.seed(42)
            if torch is not None:
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print(f"[INFO] Function executed, result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute function: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Outer test passed")
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()