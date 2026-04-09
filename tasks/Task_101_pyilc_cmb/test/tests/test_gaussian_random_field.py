import sys
import os
import dill
import numpy as np
import traceback

try:
    from agent_gaussian_random_field import gaussian_random_field
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

data_paths = ['/data/yjh/pyilc_cmb_sandbox_sandbox/run_code/std_data/standard_data_gaussian_random_field.pkl']

outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    else:
        outer_path = p

try:
    assert outer_path is not None, "No outer data file found!"
    print(f"Loaded outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'unknown')
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"  func_name: {func_name}")
    print(f"  args: {outer_args}")
    print(f"  kwargs: {outer_kwargs}")
    
    if inner_paths:
        print("Detected Scenario B: Factory/Closure pattern")
    else:
        print("Detected Scenario A: Simple function call")
    
    # The gen_data_code calls _fix_seeds_(42) at module load, which does np.random.seed(42).
    # Then gaussian_random_field is called (with seed=None), so it uses the current numpy state.
    # We must replicate that exact state. Set seed to 42 before calling.
    import random
    np.random.seed(42)
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
    
    result = gaussian_random_field(*outer_args, **outer_kwargs)
    print(f"Phase 1: gaussian_random_field returned: {type(result)}")
    
    if inner_paths:
        assert callable(result), f"Expected callable from outer call, got {type(result)}"
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output', None)
        result = result(*inner_args, **inner_kwargs)
        print(f"Phase 2: operator returned: {type(result)}")
    
    passed, msg = recursive_check(expected_output, result)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"FAIL: Verification failed")
        print(f"  Message: {msg}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)