import sys
import os
import dill
import torch
import numpy as np
import traceback

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/phiflow_pde_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl'
    ]

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

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print("Outer data loaded successfully.")
        print("Outer func_name: " + str(outer_data.get('func_name', 'N/A')))
    except Exception as e:
        print("ERROR loading outer data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute function
    try:
        # Handle device argument - ensure compatibility
        if 'device' in outer_kwargs and outer_kwargs['device'] is not None:
            dev = outer_kwargs['device']
            if isinstance(dev, torch.device):
                if dev.type == 'cuda' and not torch.cuda.is_available():
                    outer_kwargs['device'] = torch.device('cpu')
            elif isinstance(dev, str):
                if 'cuda' in dev and not torch.cuda.is_available():
                    outer_kwargs['device'] = 'cpu'

        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print("ERROR executing load_and_preprocess_data: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    # Phase 3: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        if not callable(agent_result):
            print("ERROR: Expected callable from outer function, got " + str(type(agent_result)))
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print("Inner data loaded: " + str(os.path.basename(inner_path)))
            except Exception as e:
                print("ERROR loading inner data: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print("ERROR executing operator: " + str(e))
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print("TEST FAILED for " + str(os.path.basename(inner_path)) + ": " + str(msg))
                    sys.exit(1)
                else:
                    print("Check passed for " + str(os.path.basename(inner_path)))
            except Exception as e:
                print("ERROR during verification: " + str(e))
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function")
        expected = outer_output
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print("TEST FAILED: " + str(msg))
                sys.exit(1)
        except Exception as e:
            print("ERROR during verification: " + str(e))
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()