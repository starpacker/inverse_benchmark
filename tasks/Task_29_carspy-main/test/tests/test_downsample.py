import sys
import os
import dill
import numpy as np
import traceback
from agent_downsample import downsample
from verification_utils import recursive_check

# 1. Setup Data Paths
data_paths = ['/data/yjh/carspy-main_sandbox/run_code/std_data/standard_data_downsample.pkl']

outer_path = None
inner_path = None

for path in data_paths:
    if 'parent_function' in path:
        inner_path = path
    elif 'standard_data_downsample.pkl' in path:
        outer_path = path

if outer_path is None:
    print("Error: standard_data_downsample.pkl not found in data_paths.")
    sys.exit(1)

# 2. Load Data and Execute
try:
    # Load Outer Data
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Executing downsample with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
    
    # 3. Execution Logic
    # Scenario A: Simple Function (Since downsample returns a list/array, not a callable)
    # The provided source code for 'downsample' returns 'downsampled' (a list or array), not a function.
    # Therefore, we treat this as a direct function call test.
    
    if inner_path:
        # If inner_path existed (Scenario B - Factory), we would do this:
        # agent_operator = downsample(*outer_args, **outer_kwargs)
        # with open(inner_path, 'rb') as f:
        #     inner_data = dill.load(f)
        # actual_result = agent_operator(*inner_data['args'], **inner_data['kwargs'])
        # expected_output = inner_data['output']
        pass
    else:
        # Scenario A: Direct execution
        actual_result = downsample(*outer_args, **outer_kwargs)

    # 4. Verification
    print("Verifying results...")
    is_match, failure_msg = recursive_check(expected_output, actual_result)

    if not is_match:
        print(f"Verification Failed: {failure_msg}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

except Exception as e:
    print(f"An error occurred during test execution: {e}")
    traceback.print_exc()
    sys.exit(1)