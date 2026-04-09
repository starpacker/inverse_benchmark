import sys
import os
import dill
import traceback

# Import the target function
from agent_params_from_vector import params_from_vector
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/poseidon_atm_sandbox_sandbox/run_code/std_data/standard_data_params_from_vector.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_params_from_vector.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (standard_data_params_from_vector.pkl)")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # Phase 1: Reconstruct operator
        try:
            agent_operator = params_from_vector(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling params_from_vector (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"Details: {msg}")
                sys.exit(1)

            print(f"Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        try:
            result = params_from_vector(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling params_from_vector: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed for outer data")
            print(f"Details: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()