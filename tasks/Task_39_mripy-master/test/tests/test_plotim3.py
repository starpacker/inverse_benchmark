import sys
import os
import dill
import numpy as np
import traceback
import matplotlib

# Set backend to Agg to avoid display issues during testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the directory containing the agent code to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent_plotim3 import plotim3
except ImportError:
    print("Error: Could not import 'plotim3' from 'agent_plotim3.py'. Check if the file exists and is in the path.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is missing, define a fallback or exit. 
    # Assuming it's provided in the environment.
    print("Error: Could not import 'recursive_check' from 'verification_utils.py'.")
    sys.exit(1)

def main():
    # Data paths provided by the user analysis
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_plotim3.pkl']
    
    # Identify the main data file
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif p.endswith('standard_data_plotim3.pkl'):
            outer_path = p

    if not outer_path:
        print("Error: 'standard_data_plotim3.pkl' not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract args and kwargs
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print("Executing plotim3 with loaded arguments...")
    try:
        # Run the function
        actual_result = plotim3(*args, **kwargs)
    except Exception as e:
        print(f"Error during execution of plotim3: {e}")
        traceback.print_exc()
        sys.exit(1)

    # If the function returns a callable (Factory Pattern), we need to execute the inner calls.
    # However, based on the provided code, plotim3 returns None (or closes plot).
    # But the decorator logic in the prompt suggests checking if it returns a callable.
    # Let's check if we have inner paths (Closure/Factory pattern) or just a simple execution.
    
    if inner_paths and callable(actual_result):
        print("Detected Factory/Closure pattern. Testing inner function execution...")
        for inner_p in inner_paths:
            print(f"  Loading inner data from {inner_p}...")
            try:
                with open(inner_p, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"  Error loading inner pickle: {e}")
                continue
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                inner_actual = actual_result(*inner_args, **inner_kwargs)
                passed, msg = recursive_check(inner_expected, inner_actual)
                if not passed:
                    print(f"  FAILED: Inner execution mismatch in {inner_p}")
                    print(f"  {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner execution passed for {os.path.basename(inner_p)}")
            except Exception as e:
                print(f"  Error executing inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED (Factory Pattern)")
        sys.exit(0)

    else:
        # Simple Function Pattern
        # The function plotim3 likely creates a plot and returns None, or modifies state.
        # We compare whatever it returned against the recorded output.
        print("Verifying result...")
        passed, msg = recursive_check(expected_output, actual_result)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED: Output mismatch")
            print(msg)
            sys.exit(1)

if __name__ == "__main__":
    main()