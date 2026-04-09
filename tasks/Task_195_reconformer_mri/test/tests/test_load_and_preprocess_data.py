import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/reconformer_mri_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner (parent_function) file
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print("\n[Phase 1] Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Args: {len(outer_args)} positional arguments")
        print(f"  Kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function with outer args
    try:
        print("\n[Phase 1] Executing load_and_preprocess_data with outer args...")
        agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"  Result type: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if we have inner data (Scenario B) or not (Scenario A)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n[Scenario B] Factory/Closure pattern detected")
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"\n[Phase 2] Loading inner data from: {os.path.basename(inner_path)}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"  Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Inner args: {len(inner_args)} positional arguments")
                print(f"  Inner kwargs: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("\n[Phase 2] Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Result type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                print("\n[Verification] Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed: {msg}")
                    
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("\n[Scenario A] Simple function pattern detected")
        
        result = agent_operator
        expected = outer_output
        
        # Verify results
        try:
            print("\n[Verification] Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"  Verification passed: {msg}")
                
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*50)
    print("TEST PASSED")
    print("="*50)
    sys.exit(0)


if __name__ == "__main__":
    main()