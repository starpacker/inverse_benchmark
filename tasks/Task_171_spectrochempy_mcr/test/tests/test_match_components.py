import sys
import os
import dill
import traceback

# Add necessary paths
sys.path.insert(0, '/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code')

# Import the target function
from agent_match_components import match_components
from verification_utils import recursive_check

def main():
    """Main test function for match_components."""
    
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_match_components.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if it's an inner data file (contains "parent_function")
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the outer data file (exact match pattern)
        elif basename == 'standard_data_match_components.pkl':
            outer_path = path
    
    # If no outer path found, use the first available path
    if outer_path is None and len(data_paths) > 0:
        outer_path = data_paths[0]
    
    if outer_path is None:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data for function: {outer_data.get('func_name', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function with outer data
    try:
        agent_result = match_components(*outer_args, **outer_kwargs)
        print("Successfully executed match_components with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute match_components: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Check if agent_result is callable
        if not callable(agent_result):
            print("WARNING: Result is not callable, treating as Scenario A")
            # Fall back to Scenario A
            result = agent_result
            expected = outer_output
        else:
            print("Agent operator is callable, proceeding with inner data execution")
            
            # Process each inner path
            for inner_path in inner_paths:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    print(f"Successfully loaded inner data from: {inner_path}")
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the agent operator with inner data
                try:
                    result = agent_result(*inner_args, **inner_kwargs)
                    print("Successfully executed agent operator with inner data")
                except Exception as e:
                    print(f"ERROR: Failed to execute agent operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                expected = inner_output
                
                # Verify results
                try:
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner test passed for: {inner_path}")
                except Exception as e:
                    print(f"ERROR: Verification failed: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        result = agent_result
        expected = outer_output
    
    # Final verification for Scenario A or fallback
    try:
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()