import sys
import os
import dill
import torch
import numpy as np
import traceback
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/phiflow_pde_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Loaded outer data from: " + outer_path)
    except Exception as e:
        print("ERROR loading outer data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Move tensors to available device
    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            if torch.cuda.is_available():
                return obj.cuda()
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [move_to_device(x) for x in obj]
            return type(obj)(moved)
        return obj

    outer_args = move_to_device(list(outer_args))
    outer_kwargs = move_to_device(outer_kwargs)

    # Use a temporary directory for save_dir to avoid polluting the workspace
    tmp_save_dir = tempfile.mkdtemp(prefix="test_eval_results_")
    if 'save_dir' not in outer_kwargs:
        # Check if save_dir is a positional arg (3rd argument)
        if len(outer_args) >= 3:
            outer_args[2] = tmp_save_dir
        else:
            outer_kwargs['save_dir'] = tmp_save_dir

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            print("Phase 1: Creating operator via evaluate_results...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Operator created successfully.")
        except Exception as e:
            print("ERROR in Phase 1 (creating operator): " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("ERROR: evaluate_results did not return a callable operator.")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("Loaded inner data from: " + inner_path)
            except Exception as e:
                print("ERROR loading inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            inner_args = move_to_device(list(inner_data.get('args', ())))
            inner_kwargs = move_to_device(inner_data.get('kwargs', {}))
            expected = inner_data.get('output', None)

            try:
                print("Phase 2: Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator execution completed.")
            except Exception as e:
                print("ERROR in Phase 2 (executing operator): " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print("ERROR during verification: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print("TEST FAILED: " + str(msg))
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        try:
            print("Phase 1: Running evaluate_results directly...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Function execution completed.")
        except Exception as e:
            print("ERROR running evaluate_results: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print("ERROR during verification: " + str(e))
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print("TEST FAILED: " + str(msg))
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()