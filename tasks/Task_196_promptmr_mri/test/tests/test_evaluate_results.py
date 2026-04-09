import sys
import os
import dill
import numpy as np
import traceback

# Add the run_code directory to path if needed
sys.path.insert(0, '/data/yjh/promptmr_mri_sandbox_sandbox/run_code')

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("Phase 1: Created operator successfully.")
        except Exception as e:
            print(f"ERROR in Phase 1 (creating operator): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable.")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Executed operator successfully.")
            except Exception as e:
                print(f"ERROR in Phase 2 (executing operator): {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_data.get('output')
        
        # Use a temporary output directory to avoid conflicts
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='test_evaluate_results_')
        
        # Check if output_dir is in kwargs or args
        # The function signature is: evaluate_results(data_dict, result_dict, output_dir='results')
        # Override output_dir to use temp directory
        modified_kwargs = dict(outer_kwargs)
        if 'output_dir' not in modified_kwargs:
            # Check if output_dir is provided as a positional arg (3rd arg)
            if len(outer_args) >= 3:
                outer_args = list(outer_args)
                outer_args[2] = tmp_dir
                outer_args = tuple(outer_args)
            else:
                modified_kwargs['output_dir'] = tmp_dir
        else:
            modified_kwargs['output_dir'] = tmp_dir
        
        try:
            result = evaluate_results(*outer_args, **modified_kwargs)
            print("Phase 1: Executed function successfully.")
        except Exception as e:
            print(f"ERROR executing function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    sys.exit(0)

if __name__ == '__main__':
    main()