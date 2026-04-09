import sys
import os
import dill
import numpy as np
import traceback

# Ensure matplotlib doesn't try to open display
import matplotlib
matplotlib.use('Agg')

def main():
    try:
        from agent_evaluate_results import evaluate_results
        from verification_utils import recursive_check
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    data_paths = [
        '/data/yjh/impedance_eis_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"Phase 1: evaluate_results returned type={type(agent_operator)}")
        except Exception as e:
            print(f"ERROR in Phase 1 (creating operator): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR executing operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for inner data {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for inner data {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Use a temporary output directory to avoid polluting the filesystem
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')

        # Override output_dir in kwargs if present, or in args
        # The function signature is: evaluate_results(data, inversion_result, output_dir='results')
        # We need to redirect output_dir to temp_dir
        modified_kwargs = dict(outer_kwargs)
        
        # Check if output_dir is in kwargs
        if 'output_dir' in modified_kwargs:
            modified_kwargs['output_dir'] = temp_dir
        elif len(outer_args) >= 3:
            # output_dir is the 3rd positional arg
            outer_args = list(outer_args)
            outer_args[2] = temp_dir
            outer_args = tuple(outer_args)
        else:
            modified_kwargs['output_dir'] = temp_dir

        try:
            result = evaluate_results(*outer_args, **modified_kwargs)
            print(f"evaluate_results returned type={type(result)}")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # The expected output was generated with a potentially different output_dir,
        # so we need to handle comparison carefully. The metrics dict should be
        # the same structurally except we don't worry about file paths.
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()