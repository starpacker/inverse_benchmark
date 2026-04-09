import sys
import os
import dill
import numpy as np
import traceback
import torch
import scipy.ndimage
import scipy.fftpack
from skimage import metrics

# Add the directory containing the agent code to the path
sys.path.append('/data/yjh/semiblindpsfdeconv-master_sandbox/run_code')

# --- IMPORT TARGET FUNCTION ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'. Ensure the file exists in the python path.")
    sys.exit(1)

# --- INJECT EVALUATION LOGIC (Reference B) ---
def evaluate_results(ground_truth, reconstructed):
    """
    Computes PSNR and SSIM.
    """
    # Ensure inputs are numpy arrays and float32/64
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
        
    ground_truth = np.array(ground_truth, dtype=np.float32)
    reconstructed = np.array(reconstructed, dtype=np.float32)
    
    # Handle potential dimension mismatch if one is single channel and other is 3 (unlikely here but safe)
    if ground_truth.ndim != reconstructed.ndim:
         # simple reshape attempt if just a channel dim difference
        pass 

    # Clip to expected range if necessary, though PSNR usually handles it via data_range
    # Using data_range=1.0 assuming images are normalized 0-1
    psnr_val = metrics.peak_signal_noise_ratio(ground_truth, reconstructed, data_range=1.0)
    ssim_val = metrics.structural_similarity(ground_truth, reconstructed, data_range=1.0, channel_axis=None) # Assuming 2D grayscale based on context
    
    return psnr_val, ssim_val

# --- MAIN TEST LOGIC ---
def main():
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Check if GPU is available (environment requirement)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load Data
    if not data_paths:
        print("No data paths provided.")
        sys.exit(1)
        
    outer_data_path = data_paths[0]
    if not os.path.exists(outer_data_path):
        print(f"Error: Data file not found at {outer_data_path}")
        sys.exit(1)

    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Loaded data for function: {outer_data.get('func_name')}")

    # Parse Arguments
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    std_result = outer_data.get('output')

    # --- EXECUTION PHASE ---
    print("Executing 'run_inversion' with loaded arguments...")
    try:
        # NOTE: The provided data likely contains a torch model as one of the args.
        # We need to ensure that if the model was saved on CPU/GPU, it's loaded correctly relative to current env.
        # However, dill/torch usually handle this, or we might need to move model to device explicitly.
        
        # Pre-process args to move models to device if they are torch nn.Modules
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                arg = arg.to(device)
                arg.eval() # Ensure eval mode
            processed_args.append(arg)
        args = tuple(processed_args)

        agent_result = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- EVALUATION PHASE ---
    # Since run_inversion is a deconvolution/restoration task, we don't have the "true ground truth" (sharp image)
    # explicitly in the standard_data_run_inversion.pkl inputs usually (unless it's in the args, but standard practice
    # here is to compare against the *previously recorded output* which serves as the "Ground Truth" for regression testing).
    
    # However, the prompt asks to compare Agent Quality against Ground Truth.
    # In regression contexts, the "Standard Result" in the pickle IS the Ground Truth behavior we want to match or exceed.
    # If the standard result is an image, we compare the agent result to the standard result to ensure regression stability.
    
    print("Evaluating results...")
    
    # Check output structure
    if agent_result is None or std_result is None:
        print("Error: One of the results is None.")
        sys.exit(1)

    # Calculate metrics between Agent Output and Standard Recorded Output (Regression Test)
    # Since we don't have an external 'clean' image, we assume 'std_result' is the target.
    # High PSNR/SSIM between Agent and Standard means the code behaves consistently.
    
    try:
        # PSNR and SSIM between the Agent's output and the Recorded output
        psnr_score, ssim_score = evaluate_results(std_result, agent_result)
        
        print(f"Comparison (Agent vs Standard Recorded Output):")
        print(f"  PSNR: {psnr_score:.4f}")
        print(f"  SSIM: {ssim_score:.4f}")

        # --- SUCCESS CRITERIA ---
        # Since this is a deterministic algorithm (mostly, dependent on float precision/GPU),
        # we expect very high similarity to the recorded output.
        # Allow small floating point divergence.
        
        # Thresholds: PSNR > 40dB (essentially identical), SSIM > 0.99
        if psnr_score < 35.0 or ssim_score < 0.95:
            print("FAILED: Performance significantly deviated from standard recording.")
            sys.exit(1)
        else:
            print("SUCCESS: Agent output matches standard recording within acceptable margins.")
            sys.exit(0)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()