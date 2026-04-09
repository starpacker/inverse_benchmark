import sys
import os
import dill
import traceback
import numpy as np

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
        '/data/yjh/opt_recon_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
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
        print("ERROR: No outer data file found (standard_data_evaluate_results.pkl)")
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

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Create the operator
        try:
            # Use a temporary output directory to avoid polluting existing dirs
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
            
            # Check if output_dir is in kwargs or args
            # The function signature: evaluate_results(phantom, reconstruction, sinograms_noisy, theta, params, output_dir='results')
            # Inject temp output_dir to avoid file conflicts
            modified_kwargs = dict(outer_kwargs)
            if 'output_dir' not in modified_kwargs:
                # output_dir is the 6th positional arg (index 5)
                if len(outer_args) <= 5:
                    modified_kwargs['output_dir'] = temp_dir
                # else it's already in args, we leave it
            
            agent_operator = evaluate_results(*outer_args, **modified_kwargs)
            print("Phase 1: evaluate_results returned successfully")
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
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator execution succeeded")
            except Exception as e:
                print(f"ERROR in Phase 2 (executing operator): {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        expected = outer_data.get('output')

        # Phase 1: Call the function
        try:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')

            # Modify output_dir to use temp directory
            modified_kwargs = dict(outer_kwargs)
            outer_args_list = list(outer_args)
            
            # The function signature: evaluate_results(phantom, reconstruction, sinograms_noisy, theta, params, output_dir='results')
            # output_dir is arg index 5
            if len(outer_args_list) > 5:
                # output_dir is provided as positional arg, replace it
                outer_args_list[5] = temp_dir
                result = evaluate_results(*outer_args_list, **modified_kwargs)
            elif 'output_dir' in modified_kwargs:
                modified_kwargs['output_dir'] = temp_dir
                result = evaluate_results(*outer_args_list, **modified_kwargs)
            else:
                modified_kwargs['output_dir'] = temp_dir
                result = evaluate_results(*outer_args_list, **modified_kwargs)

            print("Phase 1: evaluate_results returned successfully")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()