import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_rm_1 import rm_1
from verification_utils import recursive_check


def main():
    # Data paths provided (empty in this case, will use default pattern)
    data_paths = []
    
    # If data_paths is empty, try to find data files in standard locations
    if not data_paths:
        # Look for data files in current directory and common test data locations
        possible_locations = [
            '.',
            './data',
            './test_data',
            '../data',
            '../test_data'
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                for f in os.listdir(loc):
                    if f.endswith('.pkl') and 'rm_1' in f:
                        data_paths.append(os.path.join(loc, f))
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_rm_1.pkl' or filename.endswith('_rm_1.pkl'):
            outer_path = path
    
    # If no data files found, create a simple test case
    if outer_path is None and not inner_paths:
        print("No data files found. Running with synthetic test data...")
        
        try:
            # Create test inputs for rm_1
            # rm_1 expects (Biter, x, y) where Biter is an array and x, y are dimensions
            
            # Test case 1: x odd, y even
            x1, y1 = 5, 6
            Biter1 = np.random.randint(0, 255, (x1 + 1, y1), dtype=np.uint8)
            result1 = rm_1(Biter1, x1, y1)
            expected1 = np.zeros((x1, y1), dtype=np.uint8)
            expected1[:, :] = Biter1[0:x1, :]
            
            passed1, msg1 = recursive_check(expected1, result1)
            if not passed1:
                print(f"Test case 1 (x odd, y even) FAILED: {msg1}")
                sys.exit(1)
            print("Test case 1 (x odd, y even) PASSED")
            
            # Test case 2: x even, y odd
            x2, y2 = 6, 5
            Biter2 = np.random.randint(0, 255, (x2, y2 + 1), dtype=np.uint8)
            result2 = rm_1(Biter2, x2, y2)
            expected2 = np.zeros((x2, y2), dtype=np.uint8)
            expected2[:, :] = Biter2[:, 0:y2]
            
            passed2, msg2 = recursive_check(expected2, result2)
            if not passed2:
                print(f"Test case 2 (x even, y odd) FAILED: {msg2}")
                sys.exit(1)
            print("Test case 2 (x even, y odd) PASSED")
            
            # Test case 3: x odd, y odd
            x3, y3 = 5, 7
            Biter3 = np.random.randint(0, 255, (x3 + 1, y3 + 1), dtype=np.uint8)
            result3 = rm_1(Biter3, x3, y3)
            expected3 = np.zeros((x3, y3), dtype=np.uint8)
            expected3[:, :] = Biter3[0:x3, 0:y3]
            
            passed3, msg3 = recursive_check(expected3, result3)
            if not passed3:
                print(f"Test case 3 (x odd, y odd) FAILED: {msg3}")
                sys.exit(1)
            print("Test case 3 (x odd, y odd) PASSED")
            
            # Test case 4: x even, y even
            x4, y4 = 6, 8
            Biter4 = np.random.randint(0, 255, (x4, y4), dtype=np.uint8)
            result4 = rm_1(Biter4, x4, y4)
            expected4 = Biter4  # Should return unchanged
            
            passed4, msg4 = recursive_check(expected4, result4)
            if not passed4:
                print(f"Test case 4 (x even, y even) FAILED: {msg4}")
                sys.exit(1)
            print("Test case 4 (x even, y even) PASSED")
            
            print("TEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Error during synthetic test: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Load and test with data files
    try:
        # Phase 1: Load outer data and run function
        if outer_path and os.path.exists(outer_path):
            print(f"Loading outer data from: {outer_path}")
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            # Execute the function
            result = rm_1(*outer_args, **outer_kwargs)
            
            # Phase 2: Check for inner data (factory pattern)
            if inner_paths:
                # This is a factory/closure pattern
                agent_operator = result
                
                if not callable(agent_operator):
                    print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                    sys.exit(1)
                
                # Process each inner data file
                for inner_path in inner_paths:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output')
                    
                    # Execute the operator
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Compare results
                    passed, msg = recursive_check(expected, actual_result)
                    if not passed:
                        print(f"FAILED for {inner_path}: {msg}")
                        sys.exit(1)
                    print(f"PASSED for {inner_path}")
            else:
                # Simple function pattern - compare directly
                expected = outer_data.get('output')
                
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAILED: {msg}")
                    sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"ERROR: Outer data file not found at {outer_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()