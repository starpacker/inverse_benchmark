import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/spikeinterface_sorting_sandbox_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def compare_recordings(expected, actual, label="recording"):
    """Compare two SpikeInterface recording objects by their properties and data."""
    errors = []
    
    # Compare basic properties
    if expected.get_num_channels() != actual.get_num_channels():
        errors.append(f"{label}: num_channels mismatch: {expected.get_num_channels()} vs {actual.get_num_channels()}")
    
    if expected.get_sampling_frequency() != actual.get_sampling_frequency():
        errors.append(f"{label}: sampling_frequency mismatch: {expected.get_sampling_frequency()} vs {actual.get_sampling_frequency()}")
    
    if expected.get_num_samples() != actual.get_num_samples():
        errors.append(f"{label}: num_samples mismatch: {expected.get_num_samples()} vs {actual.get_num_samples()}")
    
    if expected.get_total_duration() != actual.get_total_duration():
        errors.append(f"{label}: duration mismatch: {expected.get_total_duration()} vs {actual.get_total_duration()}")
    
    if expected.get_num_segments() != actual.get_num_segments():
        errors.append(f"{label}: num_segments mismatch: {expected.get_num_segments()} vs {actual.get_num_segments()}")
    
    # Compare channel ids
    exp_ch = list(expected.get_channel_ids())
    act_ch = list(actual.get_channel_ids())
    if exp_ch != act_ch:
        errors.append(f"{label}: channel_ids mismatch")
    
    # Compare a small chunk of traces to verify data equivalence
    try:
        n_samples_check = min(1000, expected.get_num_samples())
        exp_traces = expected.get_traces(start_frame=0, end_frame=n_samples_check)
        act_traces = actual.get_traces(start_frame=0, end_frame=n_samples_check)
        if not np.allclose(exp_traces, act_traces, rtol=1e-5, atol=1e-6):
            max_diff = np.max(np.abs(exp_traces - act_traces))
            errors.append(f"{label}: traces data mismatch (max_diff={max_diff})")
    except Exception as e:
        errors.append(f"{label}: could not compare traces: {e}")
    
    return errors


def compare_sortings(expected, actual, label="sorting"):
    """Compare two SpikeInterface sorting objects by their properties and data."""
    errors = []
    
    exp_units = list(expected.unit_ids)
    act_units = list(actual.unit_ids)
    
    if len(exp_units) != len(act_units):
        errors.append(f"{label}: num_units mismatch: {len(exp_units)} vs {len(act_units)}")
        return errors
    
    if not np.array_equal(exp_units, act_units):
        errors.append(f"{label}: unit_ids mismatch: {exp_units} vs {act_units}")
        return errors
    
    if expected.get_sampling_frequency() != actual.get_sampling_frequency():
        errors.append(f"{label}: sampling_frequency mismatch")
    
    # Compare spike trains for each unit
    for uid in exp_units:
        exp_st = expected.get_unit_spike_train(uid)
        act_st = actual.get_unit_spike_train(uid)
        if len(exp_st) != len(act_st):
            errors.append(f"{label}: unit {uid} spike count mismatch: {len(exp_st)} vs {len(act_st)}")
        elif not np.array_equal(exp_st, act_st):
            errors.append(f"{label}: unit {uid} spike trains differ")
    
    return errors


def main():
    data_paths = ['/data/yjh/spikeinterface_sorting_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    if inner_paths:
        print("Detected Scenario B (Factory/Closure pattern)")
    else:
        print("Detected Scenario A (Simple function)")

    # Phase 1: Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Phase 1: Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        # Scenario B
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output', None)

            if not callable(result):
                print("FAIL: Expected callable from Phase 1 for Scenario B.")
                sys.exit(1)

            try:
                actual_result = result(*inner_args, **inner_kwargs)
                print("Phase 2: Inner execution successful.")
            except Exception as e:
                print(f"FAIL: Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed.\n  Message: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A - compare result with expected_output
        # The result is a tuple: (recording_cmr, recording_raw, sorting_gt)
        # Expected is also a tuple of the same structure
        # These are SpikeInterface objects that need custom comparison
        
        all_errors = []
        
        try:
            if not isinstance(result, tuple) or not isinstance(expected_output, tuple):
                print(f"FAIL: Expected tuple outputs, got result={type(result)}, expected={type(expected_output)}")
                sys.exit(1)
            
            if len(result) != len(expected_output):
                print(f"FAIL: Output tuple length mismatch: {len(result)} vs {len(expected_output)}")
                sys.exit(1)
            
            # result[0] = recording_cmr (preprocessed recording)
            # result[1] = recording_raw (raw recording)  
            # result[2] = sorting_gt (ground truth sorting)
            
            # Compare recording_cmr
            errors = compare_recordings(expected_output[0], result[0], label="recording_cmr")
            all_errors.extend(errors)
            
            # Compare recording_raw
            errors = compare_recordings(expected_output[1], result[1], label="recording_raw")
            all_errors.extend(errors)
            
            # Compare sorting_gt
            errors = compare_sortings(expected_output[2], result[2], label="sorting_gt")
            all_errors.extend(errors)
            
            if all_errors:
                print("FAIL: Verification failed.")
                for err in all_errors:
                    print(f"  - {err}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"FAIL: Custom verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()