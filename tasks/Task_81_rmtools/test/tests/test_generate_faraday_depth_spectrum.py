import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_faraday_depth_spectrum import generate_faraday_depth_spectrum

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for generate_faraday_depth_spectrum"""
    
    # Data paths provided
    data_paths = ['/data/yjh/rmtools_sandbox_sandbox/run_code/std_data/standard_data_generate_faraday_depth_spectrum.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the outer path (exact match pattern)
        elif basename == 'standard_data_generate_faraday_depth_spectrum.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_faraday_depth_spectrum.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - Function name: {outer_data.get('func_name', 'N/A')}")
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function with outer args/kwargs
    try:
        print("Executing generate_faraday_depth_spectrum with outer arguments...")
        agent_result = generate_faraday_depth_spectrum(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR: Failed to execute generate_faraday_depth_spectrum: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from generate_faraday_depth_spectrum, got {type(agent_result)}")
            sys.exit(1)
        
        print("Agent operator is callable. Proceeding with inner execution...")
        
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - Function name: {inner_data.get('func_name', 'N/A')}")
                print(f"  - Number of args: {len(inner_args)}")
                print(f"  - Number of kwargs: {len(inner_kwargs)}")
                
                # Execute the operator with inner args/kwargs
                print("Executing agent operator with inner arguments...")
                result = agent_result(*inner_args, **inner_kwargs)
                print("Inner execution completed successfully.")
                
                # Verify results
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function test")
        
        result = agent_result
        expected = outer_output
        
        # Verify results
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()