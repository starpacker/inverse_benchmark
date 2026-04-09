import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_crop_array import crop_array
from verification_utils import recursive_check


def main():
    """Main test function for crop_array."""
    
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_crop_array.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function')
        if 'parent_function' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact match pattern)
        elif basename == 'standard_data_crop_array.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer path exists
    if outer_path and not inner_paths:
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected = outer_data.get('output')
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Args count: {len(args)}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
        except Exception as e:
            print(f"ERROR loading outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            result = crop_array(*args, **kwargs)
            print("Function executed successfully")
            
        except Exception as e:
            print(f"ERROR executing crop_array: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Compare results
            passed, msg = recursive_check(expected, result)
            
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
    
    # Scenario B: Factory/Closure pattern - both outer and inner paths exist
    elif outer_path and inner_paths:
        try:
            # Phase 1: Load outer data and reconstruct operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Outer args count: {len(outer_args)}")
            print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
            
        except Exception as e:
            print(f"ERROR loading outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Create the operator/closure
            agent_operator = crop_array(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully")
            
        except Exception as e:
            print(f"ERROR creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Loaded inner data from: {inner_path}")
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Execute the operator with inner args
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully")
                
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Compare results
                passed, msg = recursive_check(expected, result)
                
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print("ERROR: Could not determine test scenario from data paths")
        print(f"Outer path: {outer_path}")
        print(f"Inner paths: {inner_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()