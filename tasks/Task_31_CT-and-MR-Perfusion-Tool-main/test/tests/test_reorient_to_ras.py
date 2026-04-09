import sys
import os
import dill
import numpy as np
import SimpleITK as sitk
import traceback
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Ensure proper path for imports
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_reorient_to_ras import reorient_to_ras
except ImportError:
    print("Error: Could not import reorient_to_ras from agent_reorient_to_ras")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import verification_utils")
    sys.exit(1)

def compare_sitk_images(img1, img2):
    """
    Robust comparison for SimpleITK images to bypass internal pointer/metadata 
    differences that might trip up standard equality checks.
    """
    if not isinstance(img1, sitk.Image) or not isinstance(img2, sitk.Image):
        return False, f"Type mismatch: {type(img1)} vs {type(img2)}"

    # 1. Size
    if img1.GetSize() != img2.GetSize():
        return False, f"Size mismatch: {img1.GetSize()} vs {img2.GetSize()}"

    # 2. Spacing
    if not np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=1e-5):
        return False, f"Spacing mismatch: {img1.GetSpacing()} vs {img2.GetSpacing()}"

    # 3. Origin
    if not np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=1e-5):
        return False, f"Origin mismatch: {img1.GetOrigin()} vs {img2.GetOrigin()}"

    # 4. Direction
    if not np.allclose(img1.GetDirection(), img2.GetDirection(), atol=1e-5):
        return False, f"Direction mismatch: {img1.GetDirection()} vs {img2.GetDirection()}"

    # 5. Pixel Data
    try:
        arr1 = sitk.GetArrayFromImage(img1)
        arr2 = sitk.GetArrayFromImage(img2)
        
        if arr1.shape != arr2.shape:
            return False, f"Array shape mismatch: {arr1.shape} vs {arr2.shape}"

        if not np.allclose(arr1, arr2, atol=1e-5):
            diff = np.abs(arr1 - arr2)
            return False, f"Pixel data mismatch. Max diff: {np.max(diff)}"
            
    except Exception as e:
        return False, f"Error comparing pixel data: {str(e)}"

    return True, "Images match"

def main():
    # Hardcoded list of paths as per context
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_reorient_to_ras.pkl']
    
    target_path = None
    for p in data_paths:
        if 'standard_data_reorient_to_ras.pkl' in p:
            target_path = p
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Test skipped: Data file not found at {target_path}")
        sys.exit(0)
        
    try:
        with open(target_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output', None)

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------
    try:
        # Scenario A: Direct function call
        actual_output = reorient_to_ras(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    passed = False
    msg = ""

    # Specific handling for SimpleITK images to avoid 'recursive_check' failures on internal pointers
    if isinstance(actual_output, sitk.Image) and isinstance(expected_output, sitk.Image):
        passed, msg = compare_sitk_images(actual_output, expected_output)
    else:
        # Fallback for other return types
        passed, msg = recursive_check(expected_output, actual_output)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()