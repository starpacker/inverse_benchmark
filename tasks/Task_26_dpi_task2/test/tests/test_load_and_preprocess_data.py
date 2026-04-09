import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data"""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_load_and_preprocess_data.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Fix random seed to match data generation
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function executed successfully, result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) and we have inner data
    if callable(agent_result) and not isinstance(agent_result, type) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern - result is callable")
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {len(inner_args)} positional arguments")
                print(f"Inner kwargs: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                print("Executing returned operator with inner arguments...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function - compare result directly with outer output
        print("Scenario A: Simple function execution")
        
        result = agent_result
        expected = outer_output
        
        # Verify results
        try:
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                # Print additional debug info
                if isinstance(expected, dict) and isinstance(result, dict):
                    print(f"Expected keys: {expected.keys()}")
                    print(f"Result keys: {result.keys()}")
                    for key in expected.keys():
                        if key in result:
                            exp_val = expected[key]
                            res_val = result[key]
                            if isinstance(exp_val, np.ndarray) and isinstance(res_val, np.ndarray):
                                print(f"Key '{key}': expected shape {exp_val.shape}, got shape {res_val.shape}")
                                if exp_val.shape == res_val.shape:
                                    diff = np.abs(exp_val - res_val)
                                    print(f"  Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()