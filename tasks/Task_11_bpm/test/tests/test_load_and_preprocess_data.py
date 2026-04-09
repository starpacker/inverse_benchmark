import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Step 2: File Logic Setup - Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    print(f"Scenario: {'B (Factory/Closure Pattern)' if is_factory_pattern else 'A (Simple Function)'}")
    
    try:
        # Step 3: Phase 1 - Load outer data and reconstruct operator
        print("\n--- Phase 1: Loading outer data ---")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("\n--- Executing load_and_preprocess_data ---")
        try:
            agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
        # Step 4: Phase 2 - Execution & Verification
        if is_factory_pattern:
            # Scenario B: Factory/Closure Pattern
            print("\n--- Phase 2: Factory Pattern - Loading inner data ---")
            
            # Verify agent_result is callable
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator, got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            # Process each inner path
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args
                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR during operator execution: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Compare results
                print("\n--- Comparing results ---")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple Function
            print("\n--- Phase 2: Simple Function - Comparing results ---")
            
            result = agent_result
            expected = outer_output
            
            if expected is None:
                print("WARNING: No expected output found in outer data")
            
            # Compare results
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"TEST PASSED: {msg}")
                sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()