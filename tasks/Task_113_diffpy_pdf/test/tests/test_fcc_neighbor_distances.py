import sys
import os
import dill
import traceback
import numpy as np

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_fcc_neighbor_distances import fcc_neighbor_distances
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/diffpy_pdf_sandbox_sandbox/run_code/std_data/standard_data_fcc_neighbor_distances.pkl'
    ]

    # ---------------------------------------------------------------
    # Step 1: Classify data files into outer (direct) and inner (parent/factory)
    # ---------------------------------------------------------------
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_fcc_neighbor_distances.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_fcc_neighbor_distances.pkl)")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Load outer data
    # ---------------------------------------------------------------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] outer_args types: {[type(a).__name__ for a in outer_args]}")
        print(f"[INFO] outer_kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 3: Determine scenario
    # ---------------------------------------------------------------
    if len(inner_paths) > 0:
        # ===================== Scenario B: Factory/Closure Pattern =====================
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        # Phase 1: Create operator
        try:
            agent_operator = fcc_neighbor_distances(*outer_args, **outer_kwargs)
            print(f"[INFO] agent_operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from outer data: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # ===================== Scenario A: Simple Function =====================
        print("[INFO] Scenario A detected: Simple function call")

        # Phase 1: Execute function
        try:
            result = fcc_neighbor_distances(*outer_args, **outer_kwargs)
            print(f"[INFO] result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute fcc_neighbor_distances: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Phase 2: Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Verification passed")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()