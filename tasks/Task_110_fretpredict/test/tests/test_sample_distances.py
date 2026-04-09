import sys
import os
import dill
import numpy as np
import traceback

# Ensure reproducibility: fix seeds before importing and running
def fix_seeds(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

from agent_sample_distances import sample_distances
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_sample_distances.pkl'
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
        print("FAIL: No outer data file found (standard_data_sample_distances.pkl)")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name')}")
        print(f"  args types: {[type(a).__name__ for a in outer_data.get('args', [])]}")
        print(f"  kwargs keys: {list(outer_data.get('kwargs', {}).keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        # Phase 1: Reconstruct operator
        try:
            fix_seeds(42)
            agent_operator = sample_distances(*outer_args, **outer_kwargs)
            print(f"Phase 1: Created operator, type={type(agent_operator).__name__}")
        except Exception as e:
            print(f"FAIL: Phase 1 failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator).__name__}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
                print(f"FAIL: Execution of operator failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Verification succeeded for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        expected = outer_data.get('output')

        # Phase 1: Execute function with same seeds
        try:
            fix_seeds(42)
            result = sample_distances(*outer_args, **outer_kwargs)
            print(f"Phase 1: Got result, type={type(result).__name__}")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Phase 2: Verify
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Message: {msg}")
                # Print some debug info
                if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
                    print(f"  Expected shape: {expected.shape}, dtype: {expected.dtype}")
                    print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
                    if expected.shape == result.shape:
                        diff = np.abs(expected - result)
                        print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean()}")
                sys.exit(1)
            else:
                print("PASS: Verification succeeded")
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()