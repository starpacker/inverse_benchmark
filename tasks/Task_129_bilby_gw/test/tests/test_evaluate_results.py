import sys
import os
import dill
import traceback
import numpy as np
import tempfile
import shutil

# Ensure the module path is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    data_paths = [
        '/data/yjh/bilby_gw_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found for evaluate_results.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data loaded successfully. func_name={outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # The function writes files to `outdir`. We need to ensure the outdir exists.
    # The outdir is the last positional arg or in kwargs.
    # Based on signature: evaluate_results(data_dict, inversion_result, injection_parameters, duration, sampling_freq, minimum_freq, outdir)
    # outdir is args[6] or kwargs['outdir']
    try:
        if 'outdir' in outer_kwargs:
            outdir = outer_kwargs['outdir']
        elif len(outer_args) >= 7:
            outdir = outer_args[6]
        else:
            outdir = None

        # Create a temporary directory for output to avoid filesystem issues
        temp_outdir = tempfile.mkdtemp(prefix='test_evaluate_results_')
        print(f"Using temporary output directory: {temp_outdir}")

        # Replace outdir in args/kwargs
        if 'outdir' in outer_kwargs:
            outer_kwargs['outdir'] = temp_outdir
        elif len(outer_args) >= 7:
            outer_args = list(outer_args)
            outer_args[6] = temp_outdir
            outer_args = tuple(outer_args)
    except Exception as e:
        print(f"WARNING: Could not set up temp outdir: {e}")
        temp_outdir = None

    # Phase 2: Determine scenario and execute
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        try:
            print("Running evaluate_results to get operator...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Operator obtained. Type: {type(agent_operator)}")
            if not callable(agent_operator):
                print(f"WARNING: Returned operator is not callable, type={type(agent_operator)}")
        except Exception as e:
            print(f"ERROR running evaluate_results (outer): {e}")
            traceback.print_exc()
            if temp_outdir and os.path.exists(temp_outdir):
                shutil.rmtree(temp_outdir, ignore_errors=True)
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                if temp_outdir and os.path.exists(temp_outdir):
                    shutil.rmtree(temp_outdir, ignore_errors=True)
                sys.exit(1)

            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                if temp_outdir and os.path.exists(temp_outdir):
                    shutil.rmtree(temp_outdir, ignore_errors=True)
                sys.exit(1)

            try:
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    if temp_outdir and os.path.exists(temp_outdir):
                        shutil.rmtree(temp_outdir, ignore_errors=True)
                    sys.exit(1)
                else:
                    print("Inner test passed.")
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                if temp_outdir and os.path.exists(temp_outdir):
                    shutil.rmtree(temp_outdir, ignore_errors=True)
                sys.exit(1)

        print("TEST PASSED")
        if temp_outdir and os.path.exists(temp_outdir):
            shutil.rmtree(temp_outdir, ignore_errors=True)
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        try:
            print("Running evaluate_results...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Function completed. Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR running evaluate_results: {e}")
            traceback.print_exc()
            if temp_outdir and os.path.exists(temp_outdir):
                shutil.rmtree(temp_outdir, ignore_errors=True)
            sys.exit(1)

        expected = outer_output
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                if temp_outdir and os.path.exists(temp_outdir):
                    shutil.rmtree(temp_outdir, ignore_errors=True)
                sys.exit(1)
            else:
                print("TEST PASSED")
                if temp_outdir and os.path.exists(temp_outdir):
                    shutil.rmtree(temp_outdir, ignore_errors=True)
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            if temp_outdir and os.path.exists(temp_outdir):
                shutil.rmtree(temp_outdir, ignore_errors=True)
            sys.exit(1)


if __name__ == '__main__':
    main()