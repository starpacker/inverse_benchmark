import sys
import os
import dill
import traceback
import tempfile
import shutil

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/fretpredict_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

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
        print("FATAL: No outer data file found (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FATAL: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # The function writes files to results_dir and working_dir.
    # We need to override these with temp directories to avoid filesystem issues
    # while still testing the function's return value (metrics dict).
    # 
    # The signature is:
    # evaluate_results(p_gt, p_recon, r_grid, h, e_edges, params, results_dir, working_dir)
    # args[6] = results_dir, args[7] = working_dir

    # Create temporary directories for file output
    temp_results_dir = tempfile.mkdtemp(prefix="test_eval_results_")
    temp_working_dir = tempfile.mkdtemp(prefix="test_eval_working_")

    try:
        # Replace results_dir and working_dir in args with temp dirs
        args_list = list(outer_args)
        
        # Check if results_dir and working_dir are in args or kwargs
        if len(args_list) >= 8:
            args_list[6] = temp_results_dir
            args_list[7] = temp_working_dir
            modified_args = tuple(args_list)
            modified_kwargs = dict(outer_kwargs)
        else:
            modified_args = tuple(args_list)
            modified_kwargs = dict(outer_kwargs)
            if 'results_dir' in modified_kwargs:
                modified_kwargs['results_dir'] = temp_results_dir
            if 'working_dir' in modified_kwargs:
                modified_kwargs['working_dir'] = temp_working_dir

        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure pattern
            print("Scenario B detected: Factory/Closure pattern")
            try:
                agent_operator = evaluate_results(*modified_args, **modified_kwargs)
                print(f"Phase 1 complete. agent_operator type: {type(agent_operator)}")
            except Exception as e:
                print(f"FATAL: Failed to execute evaluate_results (Phase 1): {e}")
                traceback.print_exc()
                sys.exit(1)

            if not callable(agent_operator):
                print(f"FATAL: Expected callable from Phase 1, got {type(agent_operator)}")
                sys.exit(1)

            # Process each inner path
            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"Loaded inner data from: {inner_path}")
                except Exception as e:
                    print(f"FATAL: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)

                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                    print(f"Phase 2 complete. Result type: {type(result)}")
                except Exception as e:
                    print(f"FATAL: Failed to execute agent_operator (Phase 2): {e}")
                    traceback.print_exc()
                    sys.exit(1)

                try:
                    passed, msg = recursive_check(expected, result)
                except Exception as e:
                    print(f"FATAL: recursive_check raised exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)

        else:
            # Scenario A: Simple function call
            print("Scenario A detected: Simple function call")
            try:
                result = evaluate_results(*modified_args, **modified_kwargs)
                print(f"Function executed. Result type: {type(result)}")
            except Exception as e:
                print(f"FATAL: Failed to execute evaluate_results: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = outer_output

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FATAL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)

    finally:
        # Clean up temp directories
        try:
            shutil.rmtree(temp_results_dir, ignore_errors=True)
            shutil.rmtree(temp_working_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()