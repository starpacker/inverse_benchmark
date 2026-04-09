import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_iter_yz import iter_yz
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = []
    
    # If no data paths provided, we need to handle this case
    if not data_paths:
        print("No data paths provided. Creating a basic functionality test.")
        
        # Create test data to verify the function works correctly
        try:
            # Create sample input data
            np.random.seed(42)
            r, n, m = 5, 6, 7
            g = np.random.randn(r, n, m).astype(np.float32)
            byz = np.random.randn(r, n, m).astype(np.float32)
            para = 0.5
            mu = 0.1
            
            # Execute the function
            result = iter_yz(g, byz, para, mu)
            
            # Verify the function returns expected structure (tuple of two arrays)
            if not isinstance(result, tuple):
                print(f"FAILED: Expected tuple output, got {type(result)}")
                sys.exit(1)
            
            if len(result) != 2:
                print(f"FAILED: Expected tuple of length 2, got {len(result)}")
                sys.exit(1)
            
            Lyz, byz_out = result
            
            # Check output types
            if not isinstance(Lyz, np.ndarray):
                print(f"FAILED: Expected Lyz to be numpy array, got {type(Lyz)}")
                sys.exit(1)
            
            if not isinstance(byz_out, np.ndarray):
                print(f"FAILED: Expected byz_out to be numpy array, got {type(byz_out)}")
                sys.exit(1)
            
            # Check output shapes match input shapes
            if Lyz.shape != g.shape:
                print(f"FAILED: Lyz shape {Lyz.shape} doesn't match input shape {g.shape}")
                sys.exit(1)
            
            if byz_out.shape != byz.shape:
                print(f"FAILED: byz_out shape {byz_out.shape} doesn't match input shape {byz.shape}")
                sys.exit(1)
            
            # Check for NaN or Inf values
            if np.any(np.isnan(Lyz)) or np.any(np.isinf(Lyz)):
                print("FAILED: Lyz contains NaN or Inf values")
                sys.exit(1)
            
            if np.any(np.isnan(byz_out)) or np.any(np.isinf(byz_out)):
                print("FAILED: byz_out contains NaN or Inf values")
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"FAILED: Exception during execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Filter data paths
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_path = path
        elif filename == 'standard_data_iter_yz.pkl' or filename.endswith('_iter_yz.pkl'):
            outer_path = path
    
    if outer_path is None:
        print("FAILED: Could not find outer data file")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the target function
        result = iter_yz(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"FAILED: Error in Phase 1 (outer execution): {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_path is not None:
        # Scenario B: Factory/Closure pattern
        try:
            # The result should be callable
            if not callable(result):
                print(f"FAILED: Expected callable from outer execution, got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            # Load inner data
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            # Execute the operator
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"FAILED: Error in Phase 2 (inner execution): {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        actual_result = result
        expected = outer_data.get('output')
    
    # Phase 3: Comparison
    try:
        passed, msg = recursive_check(expected, actual_result)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"FAILED: Error during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()