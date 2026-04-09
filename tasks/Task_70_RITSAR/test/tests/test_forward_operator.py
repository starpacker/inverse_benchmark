import sys
import os
import dill
import traceback
import numpy as np

# Add the working directory to path
sys.path.insert(0, '/data/yjh/RITSAR_sandbox_sandbox/run_code')

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/RITSAR_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run forward_operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer function name: {outer_data.get('func_name')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Number of kwargs: {len(outer_kwargs)}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute forward_operator with outer args
    try:
        print("Executing forward_operator with outer args...")
        agent_result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully")
        print(f"Result type: {type(agent_result)}")
        if hasattr(agent_result, 'shape'):
            print(f"Result shape: {agent_result.shape}")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if agent_result is callable
        if not callable(agent_result):
            print("WARNING: Result is not callable, falling back to Scenario A comparison")
            # Fall back to Scenario A
            result = agent_result
            expected = outer_output
        else:
            # Process inner data
            for inner_path in inner_paths:
                try:
                    print(f"\nLoading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output')
                    
                    print(f"Inner function name: {inner_data.get('func_name')}")
                    print(f"Number of inner args: {len(inner_args)}")
                    print(f"Number of inner kwargs: {len(inner_kwargs)}")
                    
                    # Execute the operator with inner args
                    print("Executing agent_operator with inner args...")
                    result = agent_result(*inner_args, **inner_kwargs)
                    expected = inner_output
                    
                    print(f"Inner execution completed")
                    if hasattr(result, 'shape'):
                        print(f"Inner result shape: {result.shape}")
                    
                except Exception as e:
                    print(f"ERROR processing inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function test")
        result = agent_result
        expected = outer_output
    
    # Phase 3: Comparison
    try:
        print("\nComparing results...")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        if hasattr(expected, 'shape'):
            print(f"Expected shape: {expected.shape}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print("="*50)
            print(f"Failure message: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()