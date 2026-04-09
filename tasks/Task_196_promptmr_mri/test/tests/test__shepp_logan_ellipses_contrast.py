import sys
import os
import dill
import numpy
import traceback

from agent__shepp_logan_ellipses_contrast import _shepp_logan_ellipses_contrast
from verification_utils import recursive_check

data_paths = ['/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data__shepp_logan_ellipses_contrast.pkl']

def main():
    try:
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
            print("ERROR: No outer data file found.")
            sys.exit(1)

        # Phase 1: Load outer data and reconstruct
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)

        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})

        try:
            agent_result = _shepp_logan_ellipses_contrast(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute _shepp_logan_ellipses_contrast: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Phase 2: Determine scenario
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            if not callable(agent_result):
                print("ERROR: Expected callable from outer function, but got non-callable.")
                sys.exit(1)

            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')

                try:
                    result = agent_result(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute inner call: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED (inner call from {os.path.basename(inner_path)}): {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner call from {os.path.basename(inner_path)} passed.")
        else:
            # Scenario A: Simple function
            expected = outer_data.get('output')
            result = agent_result

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()