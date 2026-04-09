import sys
import os
import dill
import numpy as np
import traceback

# Fix random seeds to match data generation
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except ImportError:
    pass

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

data_paths = [
    '/data/yjh/dm4ct_bench_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
]

def main():
    # Classify paths
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

    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"[INFO] Loaded outer data from {os.path.basename(outer_path)}")
        print(f"[INFO] outer_args types: {[type(a).__name__ for a in outer_args]}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        # Re-seed before calling to match the data generation environment
        np.random.seed(42)
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_result):
            print("FAIL: Expected callable from outer call (factory pattern), got non-callable.")
            sys.exit(1)

        for inner_path in sorted(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"[INFO] Loaded inner data from {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Inner execution error: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"[INFO] Inner test passed for {os.path.basename(inner_path)}")
    else:
        # Scenario A: Simple function
        expected = outer_output
        result = agent_result

        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()