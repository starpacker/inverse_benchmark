import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/bisip_sip_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def main():
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Load outer data
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
    
    # Check if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Phase 1: Create the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created agent_operator")
        except Exception as e:
            print(f"ERROR in Phase 1 (creating operator): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        inner_path = inner_paths[0]  # Use the first inner path
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
        
        # Execute the operator with inner args
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Phase 2: Successfully executed agent_operator with inner args")
        except Exception as e:
            print(f"ERROR in Phase 2 (executing operator): {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Execute the function directly
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("Successfully executed evaluate_results")
        except Exception as e:
            print(f"ERROR executing evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
    
    # Comparison phase
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR during comparison: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()