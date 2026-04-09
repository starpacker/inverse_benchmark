import sys
import os
import dill
import traceback
import numpy as np

from agent_feff_phase import feff_phase
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/xraylarch_sandbox_sandbox/run_code/std_data/standard_data_feff_phase.pkl']

def main():
    try:
        # Filter paths to find outer and inner data files
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_paths.append(path)
            elif basename == 'standard_data_feff_phase.pkl':
                outer_path = path
        
        if outer_path is None:
            print("ERROR: Could not find outer data file (standard_data_feff_phase.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Calling feff_phase with args={outer_args}, kwargs={outer_kwargs}")
        agent_operator = feff_phase(*outer_args, **outer_kwargs)
        
        # Phase 2: Execution & Verification
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
            
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                if not callable(agent_operator):
                    print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
                    sys.exit(1)
                
                print(f"Executing agent_operator with inner args={inner_args}, kwargs={inner_kwargs}")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {inner_path}")
        else:
            # Scenario A: Simple Function
            print("Scenario A detected: Simple function test")
            result = agent_operator
            expected = outer_data.get('output')
            
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Unexpected exception occurred")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()