import sys
import os
import dill
import torch
import numpy as np
import traceback
import tempfile
import shutil

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pnp_cassi_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    print(f"Scenario: {'B (Factory/Closure Pattern)' if is_factory_pattern else 'A (Simple Function)'}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("\n--- Phase 1: Loading outer data ---")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Create a temporary directory for output files since evaluate_results writes files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory for outputs: {temp_dir}")
    
    try:
        # Modify kwargs to use temp directory if output_dir is specified
        modified_kwargs = outer_kwargs.copy()
        if 'output_dir' in modified_kwargs:
            modified_kwargs['output_dir'] = temp_dir
        else:
            # Check if output_dir might be in args (positional)
            # Based on signature: evaluate_results(recon_img, truth, psnrs, output_dir='.')
            if len(outer_args) >= 4:
                # output_dir is the 4th positional argument
                outer_args = list(outer_args)
                outer_args[3] = temp_dir
                outer_args = tuple(outer_args)
            else:
                # Add output_dir to kwargs
                modified_kwargs['output_dir'] = temp_dir
        
        print("\n--- Phase 1: Executing evaluate_results ---")
        agent_result = evaluate_results(*outer_args, **modified_kwargs)
        print(f"Function executed successfully, result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if is_factory_pattern:
        # Scenario B: Factory/Closure Pattern
        print("\n--- Phase 2: Factory Pattern - Loading inner data ---")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                print(f"\nProcessing inner data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner data
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Compare results
                print("\n--- Comparing results ---")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    sys.exit(1)
                
                print(f"Inner test passed for: {inner_path}")
                
            except Exception as e:
                print(f"ERROR: Failed during inner data processing: {e}")
                traceback.print_exc()
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\n--- Phase 2: Simple Function - Comparing results ---")
        
        result = agent_result
        expected = outer_output
        
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")
        
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*50)
    print("TEST PASSED")
    print("="*50)
    sys.exit(0)


if __name__ == "__main__":
    main()