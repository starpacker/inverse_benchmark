import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/impedance_eis_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/impedance_eis_sandbox_sandbox')

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

data_paths = ['/data/yjh/impedance_eis_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    try:
        # Separate outer and inner paths
        outer_path = None
        inner_paths = []
        
        for p in data_paths:
            basename = os.path.basename(p)
            if 'parent_function' in basename or 'parent_' in basename:
                inner_paths.append(p)
            else:
                outer_path = p
        
        if outer_path is None:
            print("ERROR: No outer data file found.")
            sys.exit(1)
        
        # Load outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
        
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("Scenario B: Factory/Closure pattern detected.")
            
            # Phase 1: Reconstruct operator
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: forward_operator did not return a callable. Got: {type(agent_operator)}")
                sys.exit(1)
            
            print("Phase 1: Operator reconstructed successfully.")
            
            # Phase 2: Execute with inner data
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
                
                result = agent_operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for inner data {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
        else:
            # Scenario A: Simple function
            print("Scenario A: Simple function pattern detected.")
            
            # Phase 1: Execute function
            result = forward_operator(*outer_args, **outer_kwargs)
            expected = outer_output
            
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()