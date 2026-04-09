import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_get_absorption_spectra import get_absorption_spectra
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_get_absorption_spectra.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_get_absorption_spectra.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_get_absorption_spectra.pkl)")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Phase 2: Execute function
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("Detected Scenario B: Factory/Closure Pattern")
            
            # Create the operator
            agent_operator = get_absorption_spectra(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            # Load inner data and execute
            inner_path = inner_paths[0]  # Use first inner path
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data['output']
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            result = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function
            print("Detected Scenario A: Simple Function")
            
            result = get_absorption_spectra(*outer_args, **outer_kwargs)
            expected = outer_data['output']
        
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")
        
    except Exception as e:
        print(f"ERROR during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Phase 3: Comparison
        print("Comparing results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()