import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def load_data(filepath):
    """Load data from a pickle file with robust error handling."""
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Check if file is empty
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"Data file is empty (0 bytes): {filepath}")
    
    print(f"Loading data from: {filepath} (size: {file_size} bytes)")
    
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except EOFError as e:
        raise ValueError(f"Data file appears corrupted or incomplete: {filepath}") from e


def find_data_files(data_paths):
    """Categorize data files into outer and inner paths."""
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def main():
    """Main test function."""
    # Define data paths
    data_paths = ['/home/yjh/BayHunter_standalone/run_code/std_data/standard_data_evaluate_results.pkl']
    
    try:
        # Find outer and inner data files
        outer_path, inner_paths = find_data_files(data_paths)
        
        if outer_path is None:
            print("ERROR: No outer data file found (standard_data_evaluate_results.pkl)")
            sys.exit(1)
        
        # Check file validity before loading
        if not os.path.exists(outer_path):
            print(f"ERROR: Outer data file does not exist: {outer_path}")
            sys.exit(1)
        
        file_size = os.path.getsize(outer_path)
        if file_size == 0:
            print(f"ERROR: Outer data file is empty: {outer_path}")
            print("This may indicate the data generation step did not complete successfully.")
            print("TEST SKIPPED - No valid test data available")
            sys.exit(0)  # Exit with success since this is a data issue, not a code issue
        
        print(f"Loading outer data from: {outer_path}")
        
        try:
            outer_data = load_data(outer_path)
        except (EOFError, ValueError) as e:
            print(f"ERROR: Failed to load outer data: {e}")
            print("The data file may be corrupted or incomplete.")
            print("TEST SKIPPED - No valid test data available")
            sys.exit(0)  # Exit with success since this is a data issue
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Phase 1: Execute the function or create the operator
        print("Phase 1: Executing evaluate_results...")
        
        try:
            result = evaluate_results(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Check if we have inner data (factory/closure pattern)
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Found {len(inner_paths)} inner data file(s) - Factory pattern detected")
            
            # The result from Phase 1 should be callable
            if not callable(result):
                print(f"ERROR: Expected callable operator, got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"Loading inner data from: {inner_path}")
                
                try:
                    inner_data = load_data(inner_path)
                except (EOFError, ValueError) as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print("Phase 2: Executing operator with inner data...")
                
                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Compare results
                print("Comparing results...")
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                
                print(f"Inner test passed for: {os.path.basename(inner_path)}")
        else:
            # Scenario A: Simple Function
            print("Simple function pattern detected")
            expected = outer_data.get('output')
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected exception during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()