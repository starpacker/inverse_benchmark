import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_normalise_phase import normalise_phase

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for normalise_phase."""
    
    # Data paths provided
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_normalise_phase.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_normalise_phase.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find standard_data_normalise_phase.pkl")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file not found: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and run the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {len(outer_args)} arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function
        print("Executing normalise_phase with outer data...")
        result = normalise_phase(*outer_args, **outer_kwargs)
        
        # Check if we have inner paths (Scenario B: Factory/Closure pattern)
        if inner_paths:
            # Scenario B: The result should be callable
            if not callable(result):
                print("ERROR: Expected callable result for factory pattern but got non-callable")
                sys.exit(1)
            
            agent_operator = result
            print("Agent operator created successfully (callable)")
            
            # Process inner data
            for inner_path in inner_paths:
                if not os.path.exists(inner_path):
                    print(f"WARNING: Inner data file not found: {inner_path}")
                    continue
                
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Executing agent operator with inner data...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
        else:
            # Scenario A: Simple function - compare directly with outer output
            print("Scenario A: Simple function pattern detected")
            expected = outer_output
            
            # Verify results
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()