import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Could not import 'evaluate_results' from 'agent_evaluate_results'. Check file structure.")
    sys.exit(1)

from verification_utils import recursive_check

def main():
    # 1. FILE LOGIC SETUP
    # The paths provided in the prompt analysis
    data_paths = ['/data/yjh/PyHoloscope-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_path = None

    for path in data_paths:
        if path.endswith("standard_data_evaluate_results.pkl"):
            outer_path = path
        elif "parent_function_evaluate_results" in path:
            inner_path = path

    if outer_path is None:
        print("Test Skipped: No outer data file (standard_data_evaluate_results.pkl) found.")
        sys.exit(0)

    print(f"Outer Data Path: {outer_path}")
    print(f"Inner Data Path: {inner_path if inner_path else 'None (Scenario A detected)'}")

    # 2. PHASE 1: LOAD AND RUN AGENT (Outer Execution)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print("Loaded outer data successfully.")
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the function under test
        agent_result = evaluate_results(*outer_args, **outer_kwargs)
        print("Agent function executed successfully.")
        
    except Exception as e:
        print(f"Phase 1 Failed: Error during agent execution or data loading: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. PHASE 2: EXECUTION & VERIFICATION (Inner Execution or Direct Result)
    actual_result = None
    expected_result = None

    try:
        if inner_path:
            # Scenario B: The agent returned a callable (factory pattern)
            # We must now execute that callable with the inner data
            if not callable(agent_result):
                print(f"Phase 2 Error: Expected agent_result to be callable for inner execution, got {type(agent_result)}.")
                sys.exit(1)
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            # Execute the closure/operator
            actual_result = agent_result(*inner_args, **inner_kwargs)
        else:
            # Scenario A: The agent returned the final result directly
            actual_result = agent_result
            expected_result = outer_data.get('output')

    except Exception as e:
        print(f"Phase 2 Failed: Error during secondary execution or data processing: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. COMPARISON
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Comparison Logic Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()