import sys
import os
import dill
import traceback
import numpy as np

from agent_compute_psnr import compute_psnr
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_compute_psnr.pkl'
    ]

    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_compute_psnr.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find standard_data_compute_psnr.pkl")
        sys.exit(1)

    # --- Phase 1: Load outer data and run function ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    try:
        result = compute_psnr(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: compute_psnr raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Scenario A (no inner paths) ---
    if not inner_paths:
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Output mismatch: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    # --- Scenario B: Factory pattern ---
    if not callable(result):
        print(f"FAIL: Expected callable from compute_psnr, got {type(result)}")
        sys.exit(1)

    agent_operator = result

    for ip in inner_paths:
        try:
            with open(ip, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"FAIL: Could not load inner data {ip}: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        inner_expected = inner_data.get('output', None)

        try:
            inner_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"FAIL: agent_operator raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(inner_expected, inner_result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Output mismatch for {os.path.basename(ip)}: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()