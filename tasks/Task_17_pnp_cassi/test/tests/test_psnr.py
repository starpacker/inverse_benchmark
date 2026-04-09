import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_psnr import psnr
from verification_utils import recursive_check


def main():
    data_paths = ['/home/yjh/pnp_cassi_sandbox/run_code/std_data/standard_data_psnr.pkl']
    
    # Determine outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_psnr.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_psnr.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        agent_result = psnr(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute psnr with outer args/kwargs")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # agent_result should be a callable operator
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator from psnr, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner args/kwargs from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(msg)
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - result is directly comparable
        expected = outer_data.get('output')
        result = agent_result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()