import sys
import os
import dill
import traceback

# Add path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_visualize_results import visualize_results
from verification_utils import recursive_check

def main():
    """Main test function for visualize_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/rmtools_sandbox_sandbox/run_code/std_data/standard_data_visualize_results.pkl']
    
    # Analyze data paths to determine scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - args count: {len(outer_args)}")
        print(f"  - kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing visualize_results with outer args/kwargs...")
        result = visualize_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing visualize_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected factory/closure pattern with {len(inner_paths)} inner data file(s).")
        
        # Verify the result is callable
        if not callable(result):
            print(f"ERROR: Expected callable from visualize_results, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - args count: {len(inner_args)}")
                print(f"  - kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the operator with inner args
            try:
                print("Executing agent_operator with inner args/kwargs...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                print("Verifying result...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple function - result is the output to compare
        print("Detected simple function pattern (no inner data files).")
        
        expected = outer_output
        actual_result = result
        
        # Verify result
        try:
            print("Verifying result...")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()