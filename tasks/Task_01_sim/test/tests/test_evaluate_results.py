import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided (empty list means we need to find them)
    data_paths = []
    
    # If data_paths is empty, search for data files in current directory and common locations
    if not data_paths:
        search_dirs = ['.', './data', '../data', './test_data', '../test_data']
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith('.pkl') and 'evaluate_results' in f:
                        data_paths.append(os.path.join(search_dir, f))
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl' or basename.endswith('_evaluate_results.pkl'):
            outer_path = path
    
    # If no data files found, try to run a basic functionality test
    if not outer_path and not inner_paths:
        print("No data files found. Running basic functionality test...")
        
        try:
            # Create temporary test data
            import tempfile
            from skimage import io
            
            # Create a temporary directory for test files
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create test images
                test_img = np.random.rand(64, 64).astype(np.float32)
                expected_img = (test_img * 255).astype(np.uint8)
                
                expected_output_path = os.path.join(tmpdir, 'expected.png')
                output_path = os.path.join(tmpdir, 'output.png')
                
                # Save expected image
                io.imsave(expected_output_path, expected_img)
                
                # Call the function
                result = evaluate_results(
                    img_recon=test_img,
                    expected_output_path=expected_output_path,
                    output_path=output_path,
                    scaler=255.0,
                    original_dtype=np.uint8
                )
                
                # Verify result structure
                assert isinstance(result, dict), "Result should be a dictionary"
                assert 'psnr' in result, "Result should contain 'psnr'"
                assert 'ssim' in result, "Result should contain 'ssim'"
                assert 'mse' in result, "Result should contain 'mse'"
                
                print("Basic functionality test passed!")
                print(f"Result metrics: PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}, MSE={result['mse']:.6f}")
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"Basic functionality test failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    try:
        # Scenario A: Simple function test with outer data only
        if outer_path and not inner_paths:
            print(f"Loading outer data from: {outer_path}")
            
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            # Extract args and kwargs
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print("Executing evaluate_results with outer data...")
            result = evaluate_results(*outer_args, **outer_kwargs)
            
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        
        # Scenario B: Factory/Closure pattern with inner data
        elif outer_path and inner_paths:
            print(f"Loading outer data from: {outer_path}")
            
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print("Creating operator from outer data...")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            # Verify operator is callable
            if not callable(agent_operator):
                print(f"TEST FAILED: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            # Test with each inner data file
            all_passed = True
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected_output, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"Test passed for {inner_path}")
            
            if all_passed:
                print("\nTEST PASSED")
                sys.exit(0)
            else:
                sys.exit(1)
        
        # Only inner paths (unusual case, treat as simple function calls)
        elif inner_paths:
            all_passed = True
            for inner_path in inner_paths:
                print(f"Loading data from: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    data = dill.load(f)
                
                args = data.get('args', ())
                kwargs = data.get('kwargs', {})
                expected_output = data.get('output')
                
                print("Executing evaluate_results...")
                result = evaluate_results(*args, **kwargs)
                
                passed, msg = recursive_check(expected_output, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"Test passed for {inner_path}")
            
            if all_passed:
                print("\nTEST PASSED")
                sys.exit(0)
            else:
                sys.exit(1)
        
        else:
            print("No valid data files found")
            sys.exit(1)
            
    except Exception as e:
        print(f"TEST FAILED with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()