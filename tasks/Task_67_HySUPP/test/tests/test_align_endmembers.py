import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_align_endmembers import align_endmembers
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_align_endmembers.pkl']

def main():
    """Main test function for align_endmembers."""
    
    # Categorize data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_align_endmembers.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        print(f"Scenario A: Simple function test")
        print(f"Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Executing align_endmembers with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        
        try:
            actual_result = align_endmembers(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Function execution error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"FAILED: Comparison error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    
    # Scenario B: Factory/Closure pattern (has inner paths)
    elif outer_path and inner_paths:
        print(f"Scenario B: Factory/Closure pattern test")
        print(f"Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract outer args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Creating operator with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        
        try:
            agent_operator = align_endmembers(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Operator creation error: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"FAILED: Created operator is not callable, got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print(f"Executing operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Operator execution error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected_output, actual_result)
            except Exception as e:
                print(f"FAILED: Comparison error: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print(f"FAILED: Could not determine test scenario. outer_path={outer_path}, inner_paths={inner_paths}")
        sys.exit(1)

if __name__ == '__main__':
    main()