import sys
import os
import dill
import numpy as np
import traceback

from agent_make_speckle_field import make_speckle_field
from verification_utils import recursive_check

def test_make_speckle_field():
    data_paths = ['/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_make_speckle_field.pkl']

    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    expected = outer_data['output']

    # The rng object saved in args has its state AFTER the function call
    # We need to recreate the exact pre-call rng state
    # From gen_data_code: _fix_seeds_(42) is called, then np.random is seeded with 42
    # The rng passed was likely created from np.random with seed 42
    # Reconstruct by replicating the exact same setup as gen_data_code
    try:
        args = list(outer_data['args'])
        kwargs = outer_data['kwargs']

        # Recreate the rng with the same seed state as during data generation
        np.random.seed(42)
        import random
        random.seed(42)

        # The rng is args[1]; recreate it with the same seed
        # Based on standard usage, rng is likely np.random.default_rng or np.random.RandomState
        rng_original = args[1]
        if isinstance(rng_original, np.random.RandomState):
            rng_fresh = np.random.RandomState(42)
        else:
            rng_fresh = np.random.default_rng(42)

        args[1] = rng_fresh
        result = make_speckle_field(*args, **kwargs)
    except Exception as e:
        print(f"Failed to execute make_speckle_field: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"Verification error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    test_make_speckle_field()