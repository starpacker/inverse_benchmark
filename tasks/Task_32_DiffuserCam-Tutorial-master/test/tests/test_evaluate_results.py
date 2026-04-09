import sys
import os
import dill
import traceback
import shutil

# Soft import for torch to prevent immediate crash if environment lacks it
try:
    import torch
except ImportError:
    torch = None

# Soft import for numpy
try:
    import numpy as np
except ImportError:
    np = None

# Ensure local modules can be imported
sys.path.append(os.getcwd())

try:
    from agent_evaluate_results import evaluate_results
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Critical Import Error: {e}")
    sys.exit(1)

# Paths provided in the prompt
data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']

def run_test():
    # 1. Identify File Structure & Strategy
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if "parent_function" in p:
            inner_path = p
        else:
            outer_path = p

    if outer_path is None and inner_path is None:
        print("TEST FAILED: No valid data files found in provided paths.")
        sys.exit(1)

    # 2. Load Outer Data (The setup or main function call)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"TEST FAILED: Failed to load outer data from {outer_path}")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # 3. Environment Safety: Handle 'output_dir'
    # The function writes files. To avoid FileNotFoundError or writing to restricted paths,
    # we redirect output to a temporary local directory.
    safe_output_dir = "./test_results_evaluate"
    if not os.path.exists(safe_output_dir):
        os.makedirs(safe_output_dir, exist_ok=True)
    
    # Override output_dir in args/kwargs
    # Signature: evaluate_results(recon, gt, result_name, output_dir=".")
    if len(outer_args) >= 4:
        outer_args = list(outer_args)
        outer_args[3] = safe_output_dir
        outer_args = tuple(outer_args)
    else:
        # It's either in kwargs or using default. We force it in kwargs.
        outer_kwargs['output_dir'] = safe_output_dir

    print(f"Running evaluate_results with output_dir redirected to: {safe_output_dir}")

    # 4. Execution Logic
    try:
        if inner_path is None:
            # SCENARIO A: Simple Function Execution
            print("Detected Scenario A: Direct function execution.")
            
            actual_result = evaluate_results(*outer_args, **outer_kwargs)
            
            # Verify
            passed, msg = recursive_check(expected_output, actual_result)
            if passed:
                print("TEST PASSED")
                # Cleanup
                if os.path.exists(safe_output_dir):
                    shutil.rmtree(safe_output_dir)
                sys.exit(0)
            else:
                print(f"TEST FAILED: Output mismatch. {msg}")
                sys.exit(1)
        
        else:
            # SCENARIO B: Factory/Closure Pattern
            print("Detected Scenario B: Factory/Closure execution.")
            
            # Step 1: Initialize Operator
            operator = evaluate_results(*outer_args, **outer_kwargs)
            
            # Step 2: Load Inner Data
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"TEST FAILED: Failed to load inner data from {inner_path}")
                sys.exit(1)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')

            # Step 3: Execute Operator
            actual_result = operator(*inner_args, **inner_kwargs)
            
            # Step 4: Verify
            passed, msg = recursive_check(inner_expected, actual_result)
            if passed:
                print("TEST PASSED")
                if os.path.exists(safe_output_dir):
                    shutil.rmtree(safe_output_dir)
                sys.exit(0)
            else:
                print(f"TEST FAILED: Output mismatch. {msg}")
                sys.exit(1)

    except Exception as e:
        print(f"TEST FAILED: Execution crashed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()