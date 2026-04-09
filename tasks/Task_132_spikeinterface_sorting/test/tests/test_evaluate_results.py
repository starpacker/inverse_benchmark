import sys
import os
import dill
import traceback
import numpy as np

# Ensure matplotlib uses non-interactive backend before any imports that might use it
import matplotlib
matplotlib.use('Agg')

def main():
    # -------------------------------------------------------------------------
    # 1. Imports
    # -------------------------------------------------------------------------
    try:
        from agent_evaluate_results import evaluate_results
        from verification_utils import recursive_check
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Data paths analysis
    # -------------------------------------------------------------------------
    data_paths = [
        '/data/yjh/spikeinterface_sorting_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = p

    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl in data_paths.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Load outer data
    # -------------------------------------------------------------------------
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"  Outer data loaded successfully.")
        print(f"  Function name: {outer_data.get('func_name', 'N/A')}")
        print(f"  Number of args: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Determine scenario and execute
    # -------------------------------------------------------------------------
    if len(inner_paths) > 0:
        # =====================================================================
        # Scenario B: Factory/Closure Pattern
        # =====================================================================
        print("\n--- Scenario B: Factory/Closure Pattern ---")

        # Phase 1: Create the operator
        try:
            print("Phase 1: Creating operator via evaluate_results(*args, **kwargs)...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print(f"  Operator created. Type: {type(agent_operator)}")
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
                print(f"\nPhase 2: Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"  Inner data loaded. Function: {inner_data.get('func_name', 'N/A')}")
                print(f"  Number of inner args: {len(inner_args)}")
                print(f"  Inner kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("  Executing agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Compare
            try:
                print("\nPhase 3: Comparing results...")
                passed, msg = recursive_check(expected, result)
                if passed:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # =====================================================================
        # Scenario A: Simple Function
        # =====================================================================
        print("\n--- Scenario A: Simple Function ---")

        # Phase 1: Execute the function
        try:
            print("Executing evaluate_results(*args, **kwargs)...")
            # Use a temporary results directory to avoid conflicts
            # Check if results_dir is in kwargs or args
            # The function signature: evaluate_results(sorting, sorting_gt, recording_cmr, recording_raw, sorter_used, results_dir, bin_size_ms=1.0)
            # results_dir is the 6th positional arg (index 5)
            import tempfile
            temp_results_dir = None

            # We need results_dir to be writable. If the original path might not exist,
            # we create a temp directory. But we should try the original first.
            # Let's check if we can write to the specified results_dir
            if len(outer_args) > 5:
                original_results_dir = outer_args[5]
                try:
                    os.makedirs(original_results_dir, exist_ok=True)
                    # Test write permission
                    test_file = os.path.join(original_results_dir, '.write_test')
                    with open(test_file, 'w') as tf:
                        tf.write('test')
                    os.remove(test_file)
                    print(f"  Using original results_dir: {original_results_dir}")
                except (OSError, PermissionError):
                    temp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
                    outer_args = list(outer_args)
                    outer_args[5] = temp_results_dir
                    outer_args = tuple(outer_args)
                    print(f"  Using temp results_dir: {temp_results_dir}")
            elif 'results_dir' in outer_kwargs:
                original_results_dir = outer_kwargs['results_dir']
                try:
                    os.makedirs(original_results_dir, exist_ok=True)
                    test_file = os.path.join(original_results_dir, '.write_test')
                    with open(test_file, 'w') as tf:
                        tf.write('test')
                    os.remove(test_file)
                    print(f"  Using original results_dir: {original_results_dir}")
                except (OSError, PermissionError):
                    temp_results_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
                    outer_kwargs = dict(outer_kwargs)
                    outer_kwargs['results_dir'] = temp_results_dir
                    print(f"  Using temp results_dir: {temp_results_dir}")

            result = evaluate_results(*outer_args, **outer_kwargs)
            expected = outer_output
            print(f"  Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"  Result keys: {list(result.keys())}")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Phase 2: Compare
        try:
            print("\nComparing results...")
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()