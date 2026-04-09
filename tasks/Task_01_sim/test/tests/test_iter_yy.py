import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_iter_yy import iter_yy
from verification_utils import recursive_check

# Define the helper functions needed for verification
def forward_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])]
    return -out

def back_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out

def shrink(x, L):
    s = np.abs(x)
    xs = np.sign(x) * np.maximum(s - 1 / L, 0)
    return xs

def reference_iter_yy(g, byy, para, mu):
    """Reference implementation for verification"""
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy

def run_test():
    """Run tests for iter_yy function"""
    
    # Test Case 1: Basic functionality with small arrays
    print("Running Test Case 1: Basic functionality...")
    try:
        np.random.seed(42)
        # Create 3D arrays (r, n, m) shape
        g = np.random.randn(2, 4, 4).astype(np.float32)
        byy = np.random.randn(2, 4, 4).astype(np.float32)
        para = 0.5
        mu = 1.0
        
        # Get expected result from reference implementation
        expected_Lyy, expected_byy = reference_iter_yy(g, byy.copy(), para, mu)
        
        # Get actual result from target function
        actual_Lyy, actual_byy = iter_yy(g, byy.copy(), para, mu)
        
        # Verify Lyy
        passed_Lyy, msg_Lyy = recursive_check(expected_Lyy, actual_Lyy)
        if not passed_Lyy:
            print(f"Test Case 1 FAILED for Lyy: {msg_Lyy}")
            return False
            
        # Verify byy
        passed_byy, msg_byy = recursive_check(expected_byy, actual_byy)
        if not passed_byy:
            print(f"Test Case 1 FAILED for byy: {msg_byy}")
            return False
            
        print("Test Case 1 PASSED")
        
    except Exception as e:
        print(f"Test Case 1 FAILED with exception: {e}")
        traceback.print_exc()
        return False
    
    # Test Case 2: Different parameters
    print("Running Test Case 2: Different parameters...")
    try:
        np.random.seed(123)
        g = np.random.randn(3, 5, 6).astype(np.float32)
        byy = np.zeros((3, 5, 6), dtype=np.float32)  # Zero initialization
        para = 0.1
        mu = 2.0
        
        expected_Lyy, expected_byy = reference_iter_yy(g, byy.copy(), para, mu)
        actual_Lyy, actual_byy = iter_yy(g, byy.copy(), para, mu)
        
        passed_Lyy, msg_Lyy = recursive_check(expected_Lyy, actual_Lyy)
        if not passed_Lyy:
            print(f"Test Case 2 FAILED for Lyy: {msg_Lyy}")
            return False
            
        passed_byy, msg_byy = recursive_check(expected_byy, actual_byy)
        if not passed_byy:
            print(f"Test Case 2 FAILED for byy: {msg_byy}")
            return False
            
        print("Test Case 2 PASSED")
        
    except Exception as e:
        print(f"Test Case 2 FAILED with exception: {e}")
        traceback.print_exc()
        return False
    
    # Test Case 3: Edge case with larger mu (shrink behavior)
    print("Running Test Case 3: Edge case with shrink behavior...")
    try:
        np.random.seed(456)
        g = np.random.randn(2, 3, 3).astype(np.float32) * 0.1  # Small values
        byy = np.random.randn(2, 3, 3).astype(np.float32) * 0.1
        para = 1.0
        mu = 0.5  # Smaller mu means larger shrinkage threshold
        
        expected_Lyy, expected_byy = reference_iter_yy(g, byy.copy(), para, mu)
        actual_Lyy, actual_byy = iter_yy(g, byy.copy(), para, mu)
        
        passed_Lyy, msg_Lyy = recursive_check(expected_Lyy, actual_Lyy)
        if not passed_Lyy:
            print(f"Test Case 3 FAILED for Lyy: {msg_Lyy}")
            return False
            
        passed_byy, msg_byy = recursive_check(expected_byy, actual_byy)
        if not passed_byy:
            print(f"Test Case 3 FAILED for byy: {msg_byy}")
            return False
            
        print("Test Case 3 PASSED")
        
    except Exception as e:
        print(f"Test Case 3 FAILED with exception: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"Test execution failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)