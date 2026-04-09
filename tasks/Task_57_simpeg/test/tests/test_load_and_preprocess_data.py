import sys
import os
import dill
import traceback

# Add the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze data files to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner (parent_function) file or outer file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("[ERROR] Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print("[INFO] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"[INFO] Outer args: {len(outer_args)} positional arguments")
        print(f"[INFO] Outer kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function to get the operator/result
    try:
        print("[INFO] Executing load_and_preprocess_data...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Function executed successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")
        
        # Verify that agent_result is callable
        if not callable(agent_result):
            print(f"[ERROR] Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"[INFO] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"[INFO] Inner args: {len(inner_args)} positional arguments")
                print(f"[INFO] Inner kwargs: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner arguments
                print("[INFO] Executing operator with inner arguments...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                print("[INFO] Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"[FAIL] Verification failed for {inner_path}")
                    print(f"[FAIL] Message: {msg}")
                    all_passed = False
                else:
                    print(f"[PASS] Verification passed for {inner_path}")
                    
            except Exception as e:
                print(f"[ERROR] Failed processing inner data {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)
    
    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function")
        
        # The result from Phase 1 is the actual result to compare
        result = agent_result
        expected = outer_output
        
        # Compare results
        try:
            print("[INFO] Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"[FAIL] Verification failed")
                print(f"[FAIL] Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"[ERROR] Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()