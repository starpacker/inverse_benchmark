import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = [
        '/data/yjh/dm4ct_bench_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

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
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR running evaluate_results (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR loading inner data {ip}: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED on inner data {os.path.basename(ip)}: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR running evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()