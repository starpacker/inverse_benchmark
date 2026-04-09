import sys
import os
import dill
import traceback
import tempfile
import shutil

import numpy as np

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Classify paths
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

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Scenario A: simple function call (no inner paths)
    if len(inner_paths) == 0:
        print("Scenario A: Simple function call.")

        # Use a temporary directory for results_dir to avoid polluting the workspace
        tmp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
        try:
            # Override results_dir to temp directory
            if 'results_dir' in outer_kwargs:
                outer_kwargs['results_dir'] = tmp_dir
            else:
                # Check if results_dir is positional (10th arg, index 9)
                # Signature: sky_gt, recon, dirty, u_unique, v_unique, ui,
                #            lambda_l1, lambda_tsv, max_iter, results_dir='results'
                if len(outer_args) > 9:
                    outer_args = list(outer_args)
                    outer_args[9] = tmp_dir
                    outer_args = tuple(outer_args)
                else:
                    outer_kwargs['results_dir'] = tmp_dir

            result = evaluate_results(*outer_args, **outer_kwargs)
            expected = outer_output

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    else:
        # Scenario B: Factory/Closure pattern
        print("Scenario B: Factory/Closure pattern.")
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in sorted(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()