import sys
import os
import dill
from agent__normalize_constraint_params import _normalize_constraint_params
from verification_utils import recursive_check

def main():
    # 1. Load Data
    data_path = r'/home/yjh/ad_pty/code_2/run_code/std_data/standard_data__normalize_constraint_params.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    try:
        with open(data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data with dill: {e}")
        sys.exit(1)

    # 2. Extract Inputs and Expected Outputs
    try:
        args = data['args']
        kwargs = data['kwargs']
        expected_output = data['output']
    except KeyError as e:
        print(f"Error: Data file missing key {e}")
        sys.exit(1)

    # 3. Execution
    try:
        actual_output = _normalize_constraint_params(*args, **kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    passed, msg = recursive_check(expected_output, actual_output)

    # 5. Reporting
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()