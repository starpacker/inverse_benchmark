import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_q_sample import q_sample
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/dps_diffusion_sandbox_sandbox/run_code/std_data/standard_data_q_sample.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_q_sample.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_q_sample.pkl)")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    # Check if this is Scenario A (simple function) or Scenario B (factory pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Factory/Closure Pattern (Scenario B)")
        
        try:
            # Phase 1: Create the operator
            agent_operator = q_sample(*outer_args, **outer_kwargs)
            print("Successfully created operator from q_sample")
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR executing q_sample to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed operator")
            except Exception as e:
                print(f"ERROR executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print(f"Inner test passed for: {inner_path}")
    
    else:
        # Scenario A: Simple Function
        print("Detected Simple Function Pattern (Scenario A)")
        
        # The function q_sample returns (noisy_x, noise)
        # If noise was None during capture, it was randomly generated
        # The captured output[1] IS that noise, so we need to pass it explicitly
        
        try:
            # Check if expected_output is a tuple and extract the noise
            if isinstance(expected_output, tuple) and len(expected_output) == 2:
                captured_noise = expected_output[1]
                
                # Check if 'noise' was None in the original call
                # We need to pass the captured noise to get deterministic results
                if 'noise' not in outer_kwargs or outer_kwargs.get('noise') is None:
                    # Pass the captured noise explicitly
                    modified_kwargs = dict(outer_kwargs)
                    modified_kwargs['noise'] = captured_noise
                    result = q_sample(*outer_args, **modified_kwargs)
                else:
                    result = q_sample(*outer_args, **outer_kwargs)
            else:
                result = q_sample(*outer_args, **outer_kwargs)
            
            print("Successfully executed q_sample")
        except Exception as e:
            print(f"ERROR executing q_sample: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify results
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()