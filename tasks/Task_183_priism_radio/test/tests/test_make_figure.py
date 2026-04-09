import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_make_figure import make_figure
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_make_figure.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_figure.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_make_figure.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    # Check if this is a factory pattern (inner paths exist) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Factory/Closure Pattern (inner data exists)")
        
        try:
            # Run make_figure to get the operator/closure
            agent_operator = make_figure(*outer_args, **outer_kwargs)
            print("Successfully created agent operator from make_figure")
        except Exception as e:
            print(f"ERROR creating agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent operator with inner data")
            except Exception as e:
                print(f"ERROR executing agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Simple Function Pattern (no inner data)")
        
        try:
            result = make_figure(*outer_args, **outer_kwargs)
            print("Successfully executed make_figure")
        except Exception as e:
            print(f"ERROR executing make_figure: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()