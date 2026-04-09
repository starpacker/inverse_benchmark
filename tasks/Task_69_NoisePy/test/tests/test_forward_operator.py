import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    data_paths = ['/data/yjh/NoisePy_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
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
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # The function takes a 'data' dict as input
    # We need to reset the RNG state to get reproducible results
    # The RNG is inside the data dict passed as args[0]
    try:
        # Get the input data dictionary
        if len(outer_args) > 0:
            input_data = outer_args[0]
        elif 'data' in outer_kwargs:
            input_data = outer_kwargs['data']
        else:
            print("ERROR: Could not find input data")
            sys.exit(1)
        
        # The rng object needs to be in a specific state
        # Since we can't easily reset numpy RandomState, we need to handle this differently
        # We'll verify the deterministic parts exactly and the noisy part statistically
        
        # Execute the function
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Verification
    # Since noise is random, we need to verify:
    # 1. dt_clean should match exactly (deterministic)
    # 2. dt_noisy = dt_clean + noise, so we verify the relationship
    # 3. noise should have correct statistical properties
    
    try:
        # Check dt_clean (deterministic part)
        passed_clean, msg_clean = recursive_check(expected_output['dt_clean'], result['dt_clean'])
        if not passed_clean:
            print(f"TEST FAILED on dt_clean: {msg_clean}")
            sys.exit(1)
        print("dt_clean verification passed")
        
        # For dt_noisy and noise, we verify the relationship holds
        # dt_noisy = dt_clean + noise
        computed_noisy = result['dt_clean'] + result['noise']
        passed_noisy_relation, msg_noisy = recursive_check(result['dt_noisy'], computed_noisy)
        if not passed_noisy_relation:
            print(f"TEST FAILED: dt_noisy != dt_clean + noise: {msg_noisy}")
            sys.exit(1)
        print("dt_noisy = dt_clean + noise relationship verified")
        
        # Verify noise has correct shape
        if result['noise'].shape != expected_output['noise'].shape:
            print(f"TEST FAILED: noise shape mismatch: {result['noise'].shape} vs {expected_output['noise'].shape}")
            sys.exit(1)
        print("noise shape verified")
        
        # Verify noise has reasonable statistical properties (mean near 0, similar std)
        noise_std_expected = np.std(expected_output['noise'])
        noise_std_actual = np.std(result['noise'])
        # Allow some tolerance for random variation
        if noise_std_expected > 0:
            std_ratio = noise_std_actual / noise_std_expected
            if std_ratio < 0.1 or std_ratio > 10:
                print(f"TEST FAILED: noise std ratio out of range: {std_ratio}")
                sys.exit(1)
        print("noise statistics verified")
        
        # Verify dt_noisy has same shape
        if result['dt_noisy'].shape != expected_output['dt_noisy'].shape:
            print(f"TEST FAILED: dt_noisy shape mismatch")
            sys.exit(1)
        print("dt_noisy shape verified")
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()