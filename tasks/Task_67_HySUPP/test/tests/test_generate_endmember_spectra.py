import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/HySUPP_sandbox_sandbox/run_code')

from agent_generate_endmember_spectra import generate_endmember_spectra
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_generate_endmember_spectra.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_endmember_spectra.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data
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
    
    # The function uses rng which is passed as an argument
    # We need to recreate the same RNG state that was used when capturing the data
    # The rng is passed as the third argument (n_bands, n_end, rng)
    
    # Check if rng is in the args - it should be the 3rd argument
    # We need to reset the RNG to the same state as when data was captured
    
    try:
        # The original data capture used seed 42, so we need to recreate that state
        # Looking at gen_data_code: _fix_seeds_(42) was called before execution
        # The rng in args should be a numpy RandomState or Generator
        
        # Extract the arguments
        n_bands = outer_args[0]
        n_end = outer_args[1]
        original_rng = outer_args[2]
        
        # The rng was created with seed 42 in the original capture
        # We need to create a fresh rng with the same seed
        # Based on _fix_seeds_(42) in gen_data_code, numpy random was seeded with 42
        
        # Create a new RNG with seed 42 to match the original state
        # The original code used np.random.seed(42) before the function was called
        np.random.seed(42)
        
        # Check what type the rng is
        if hasattr(original_rng, 'bit_generator'):
            # It's a numpy Generator
            rng = np.random.default_rng(42)
        else:
            # It's a numpy RandomState
            rng = np.random.RandomState(42)
        
        # Recreate args with fresh rng
        new_args = (n_bands, n_end, rng)
        
        result = generate_endmember_spectra(*new_args, **outer_kwargs)
        print("Successfully executed generate_endmember_spectra")
        
    except Exception as e:
        print(f"ERROR executing generate_endmember_spectra: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check for inner paths (factory pattern)
    if inner_paths:
        print("Detected factory/closure pattern")
        # Handle inner execution
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                if callable(result):
                    result = result(*inner_args, **inner_kwargs)
                    print("Successfully executed inner function")
                    
            except Exception as e:
                print(f"ERROR with inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        print("Detected simple function pattern")
    
    # Phase 3: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()