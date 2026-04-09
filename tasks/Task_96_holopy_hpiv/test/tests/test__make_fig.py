import sys
import os
import dill
import numpy as np
import traceback
import tempfile

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use("Agg")

from agent__make_fig import _make_fig
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/holopy_hpiv_sandbox_sandbox/run_code/std_data/standard_data__make_fig.pkl'
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
        print("FAIL: No outer data file found.")
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
    outer_output = outer_data.get('output', None)

    # The function _make_fig saves a figure to a path and returns None.
    # We need to handle the 'path' argument carefully - use a temp file
    # so we don't pollute the filesystem or fail on missing directories.

    # Inspect args to find and potentially replace the path argument
    # Based on the function signature: _make_fig(holo, gt, det, mg, md, gv, pixel_size, path)
    # path is the 8th positional argument (index 7)
    try:
        # Replace path with a temporary file path to avoid filesystem issues
        args_list = list(outer_args)
        if len(args_list) >= 8:
            original_path = args_list[7]
            # Create a temp file for the output
            tmp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(tmp_dir, "test_output.png")
            args_list[7] = temp_path
            print(f"  Replaced path '{original_path}' with temp path '{temp_path}'")
        elif 'path' in outer_kwargs:
            original_path = outer_kwargs['path']
            tmp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(tmp_dir, "test_output.png")
            outer_kwargs = dict(outer_kwargs)
            outer_kwargs['path'] = temp_path
            print(f"  Replaced kwargs path '{original_path}' with temp path '{temp_path}'")
        else:
            temp_path = None
            tmp_dir = None

        outer_args = tuple(args_list)
    except Exception as e:
        print(f"WARNING: Could not replace path argument: {e}")
        temp_path = None
        tmp_dir = None

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        try:
            agent_operator = _make_fig(*outer_args, **outer_kwargs)
            print("  Phase 1: _make_fig executed, got operator.")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("  Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call")

        try:
            result = _make_fig(*outer_args, **outer_kwargs)
            print("  _make_fig executed successfully.")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Additional check: if the function is supposed to save a file, verify it exists
        if temp_path is not None and os.path.exists(temp_path):
            print(f"  Output file created at: {temp_path} (size: {os.path.getsize(temp_path)} bytes)")

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp files
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()