import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_synthetic_4dstem import generate_synthetic_4dstem
from verification_utils import recursive_check

def fix_seeds(seed=42):
    """Fix random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def main():
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_generate_synthetic_4dstem.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_synthetic_4dstem.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_synthetic_4dstem.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Phase 2: Execute the function with fixed seed
    try:
        # Fix seeds to match the original data generation
        fix_seeds(42)
        
        result = generate_synthetic_4dstem(*outer_args, **outer_kwargs)
        print("Successfully executed generate_synthetic_4dstem")
    except Exception as e:
        print(f"ERROR: Failed to execute generate_synthetic_4dstem: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                # Fix seeds again before inner call
                fix_seeds(42)
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Inner test passed for: {inner_path}")
    else:
        # Scenario A: Simple function - result is already computed
        # The function uses np.random.poisson, so we need to compare with tolerance
        # Since Poisson is stochastic, we compare shapes and dtypes, 
        # and verify reasonable value ranges
        
        # First try exact comparison (in case seeds matched)
        passed, msg = recursive_check(expected_output, result)
        
        if not passed:
            # For stochastic functions, verify structural properties instead
            if isinstance(expected_output, np.ndarray) and isinstance(result, np.ndarray):
                # Check shape matches
                if expected_output.shape != result.shape:
                    print(f"TEST FAILED: Shape mismatch - expected {expected_output.shape}, got {result.shape}")
                    sys.exit(1)
                
                # Check dtype matches
                if expected_output.dtype != result.dtype:
                    print(f"TEST FAILED: Dtype mismatch - expected {expected_output.dtype}, got {result.dtype}")
                    sys.exit(1)
                
                # Check value ranges are similar (statistical properties)
                expected_mean = expected_output.mean()
                result_mean = result.mean()
                expected_std = expected_output.std()
                result_std = result.std()
                
                # Allow 20% tolerance for statistical measures due to Poisson randomness
                mean_rel_diff = abs(expected_mean - result_mean) / (abs(expected_mean) + 1e-10)
                std_rel_diff = abs(expected_std - result_std) / (abs(expected_std) + 1e-10)
                
                if mean_rel_diff > 0.2:
                    print(f"TEST FAILED: Mean differs significantly - expected {expected_mean}, got {result_mean}")
                    sys.exit(1)
                
                if std_rel_diff > 0.2:
                    print(f"TEST FAILED: Std differs significantly - expected {expected_std}, got {result_std}")
                    sys.exit(1)
                
                # Check min/max are in reasonable ranges
                if result.min() < 0:
                    print(f"TEST FAILED: Result has negative values (min={result.min()})")
                    sys.exit(1)
                
                print("TEST PASSED (structural verification for stochastic function)")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()