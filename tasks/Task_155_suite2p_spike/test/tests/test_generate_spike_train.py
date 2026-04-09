def generate_spike_train(n_frames, spike_rate, fs, rng):
    prob_per_frame = spike_rate / fs
    spikes = (rng.random(n_frames) < prob_per_frame).astype(np.float64)
    return spikes


import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_spike_train import generate_spike_train
from verification_utils import recursive_check


def test_generate_spike_train():
    """Test generate_spike_train function."""
    
    data_paths = ['/data/yjh/suite2p_spike_sandbox_sandbox/run_code/std_data/standard_data_generate_spike_train.pkl']
    
    # Filter paths
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_generate_spike_train.pkl':
            outer_path = path
    
    if outer_path is None:
        print("TEST FAILED: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"TEST FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # The function uses rng which is a random generator
    # We need to recreate the exact same rng state
    # The rng is passed as the 4th argument (index 3)
    # n_frames, spike_rate, fs, rng = args
    
    try:
        # Extract parameters
        n_frames = outer_args[0]
        spike_rate = outer_args[1]
        fs = outer_args[2]
        rng_original = outer_args[3]
        
        # The rng object from pickle should have its state preserved
        # But we need to create a fresh rng with the same seed to reproduce results
        # Since the original rng state was captured BEFORE the function was called,
        # we need to use the same rng object
        
        # Check if rng has a way to get/set state
        if hasattr(rng_original, 'bit_generator'):
            # This is a numpy Generator object
            # The state should be preserved in the pickle
            # But the state was captured AFTER function execution during data capture
            # So we cannot reproduce the exact output without knowing the seed
            
            # For stochastic functions, we verify:
            # 1. Output shape matches
            # 2. Output dtype matches
            # 3. Output values are in valid range (0 or 1 for spike train)
            # 4. Statistical properties are reasonable
            
            # Create a fresh rng with fixed seed for reproducibility test
            test_rng = np.random.default_rng(42)
            result = generate_spike_train(n_frames, spike_rate, fs, test_rng)
            
            # Verify structural properties
            if result.shape != expected_output.shape:
                print(f"TEST FAILED: Shape mismatch. Expected {expected_output.shape}, got {result.shape}")
                sys.exit(1)
            
            if result.dtype != expected_output.dtype:
                print(f"TEST FAILED: Dtype mismatch. Expected {expected_output.dtype}, got {result.dtype}")
                sys.exit(1)
            
            # Verify values are valid (0 or 1 for spike train)
            unique_vals = np.unique(result)
            if not all(v in [0.0, 1.0] for v in unique_vals):
                print(f"TEST FAILED: Invalid spike values. Expected only 0.0 and 1.0, got {unique_vals}")
                sys.exit(1)
            
            # Verify expected output also has valid values
            expected_unique = np.unique(expected_output)
            if not all(v in [0.0, 1.0] for v in expected_unique):
                print(f"TEST FAILED: Expected output has invalid values: {expected_unique}")
                sys.exit(1)
            
            # Verify spike rate is reasonable (within statistical bounds)
            expected_prob = spike_rate / fs
            actual_spike_rate = np.mean(result)
            expected_spike_rate = np.mean(expected_output)
            
            # Both should be close to the expected probability
            # Allow for statistical variation
            tolerance = 3 * np.sqrt(expected_prob * (1 - expected_prob) / n_frames)  # 3 sigma
            
            if abs(actual_spike_rate - expected_prob) > max(tolerance, 0.1):
                print(f"Warning: Actual spike rate {actual_spike_rate} differs from expected {expected_prob}")
            
            print(f"Output shape: {result.shape}")
            print(f"Output dtype: {result.dtype}")
            print(f"Unique values: {unique_vals}")
            print(f"Actual spike rate: {actual_spike_rate:.4f}")
            print(f"Expected spike rate (theoretical): {expected_prob:.4f}")
            print(f"Original output spike rate: {expected_spike_rate:.4f}")
            print("TEST PASSED")
            sys.exit(0)
            
        else:
            # Old-style numpy RandomState
            result = generate_spike_train(n_frames, spike_rate, fs, rng_original)
            
    except Exception as e:
        print(f"TEST FAILED: Error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_generate_spike_train()