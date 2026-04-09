import sys
import os
import dill
import torch
import numpy as np
import traceback
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 1. SETUP & INJECTIONS ---

# Define device globally so unpickled objects can find it if they rely on global scope
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Could not import run_inversion from agent_run_inversion.py")
    sys.exit(1)

# Inject Global Helpers needed for unpickling custom objects (dill requirement)
# We define placeholders for classes that might be referenced by the pickle but not fully defined
# However, since we import 'run_inversion', the classes defined in 'agent_run_inversion.py' 
# (like SystemMatrix, etc.) should be available. 
# We explicitly set 'device' in the main module to satisfy the specific NameError.

# --- 2. REFEREE FUNCTION ---

def evaluate_results(gt_object, projections_noisy, recon, shape, save_name="result"):
    """
    Computes metrics (PSNR, SSIM) and saves comparison plot.
    """
    print(f"\n=== Evaluation ({save_name}) ===")
    
    # Detach and move to CPU for numpy operations
    gt_np = gt_object.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()
    proj_np = projections_noisy.detach().cpu().numpy()
    
    # Normalize for metric calculation
    # Avoid division by zero
    gt_range = gt_np.max() - gt_np.min()
    if gt_range == 0: gt_range = 1.0
    gt_norm = (gt_np - gt_np.min()) / gt_range
    
    recon_range = recon_np.max() - recon_np.min()
    if recon_range == 0: recon_range = 1.0
    recon_norm = (recon_np - recon_np.min()) / recon_range
    
    # Slice for metrics (middle slice of the Z dimension)
    mid_z = shape[2] // 2
    gt_slice = gt_norm[:, :, mid_z]
    recon_slice = recon_norm[:, :, mid_z]
    
    p = psnr(gt_slice, recon_slice, data_range=1.0)
    s = ssim(gt_slice, recon_slice, data_range=1.0)
    
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")
    
    # Save Results
    plot_filename = f"reconstruction_{save_name}.png"
    print(f"\nSaving results to '{plot_filename}'...")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(gt_slice, cmap='gray')
    ax[0].set_title("Ground Truth (Z-slice)")
    
    # Handle projections shape variations for plotting
    if len(proj_np.shape) == 3:
        proj_disp = proj_np[0, :, :].T
    else:
        proj_disp = proj_np[:, :].T
        
    ax[1].imshow(proj_disp, cmap='gray') 
    ax[1].set_title("Projection (Sample)")
    ax[2].imshow(recon_slice, cmap='gray')
    ax[2].set_title(f"OSEM Recon\nPSNR: {p:.2f}")
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print("Done.")
    
    return p, s

# --- 3. TEST LOGIC ---

def run_test():
    # Define data paths
    data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    if not data_paths:
        print("No data paths provided.")
        sys.exit(1)

    path = data_paths[0]
    print(f"Loading Data: {path}")

    try:
        with open(path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract args and output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')
    
    # IMPORTANT: The pickled 'SystemMatrix' object in 'args[0]' likely has methods
    # that reference 'device' globally. We must ensure the object's module context 
    # has 'device'.
    # Sometimes objects carry their module reference. If the object was created in a module
    # where 'device' was global, unpickling it might try to look up 'device' in that module's 
    # namespace or the current one.
    
    # We patch the SystemMatrix instance to ensure it uses the correct device if possible,
    # or ensure the global 'device' is available (which we did at the top).
    
    # The error "NameError: name 'device' is not defined" happened inside 'set_n_subsets' 
    # of the SystemMatrix class.
    
    # 1. Retrieve inputs
    system_matrix = args[0]
    projections_noisy = args[1]
    
    # 2. Get Ground Truth Object (which is usually what SystemMatrix operates on)
    # The Standard Output is the reconstructed object.
    # To evaluate effectively, we treat the 'expected_output' (Standard Run Result) 
    # as our "Ground Truth" for regression testing, OR we try to find the actual Phantom 
    # if it's hidden in the system matrix metadata.
    # Here, we will compare Agent Output vs Standard Output (Regression Test).
    # Since we don't have the "True Phantom" source separate from the reconstruction,
    # we use the Standard Output as the reference.
    
    print("Running 'run_inversion' with Agent...")
    try:
        agent_recon = run_inversion(*args, **kwargs)
    except Exception as e:
        print("Agent execution failed:")
        traceback.print_exc()
        sys.exit(1)

    # 3. Evaluation
    # Since 'run_inversion' returns the reconstructed object tensor.
    # And 'expected_output' is the reconstructed object tensor from the standard run.
    # We compare them. 
    
    # Extract shape from metadata for slicing
    try:
        # system_matrix.object_meta is usually available
        shape = system_matrix.object_meta.shape
    except:
        # Fallback shape inference
        shape = agent_recon.shape

    print("\n--- Evaluating Agent Output vs Standard Output (Regression) ---")
    # We pass 'expected_output' as GT because we want to see if Agent matches Standard.
    # Note: 'projections_noisy' is just for visualization context.
    
    p_agent, s_agent = evaluate_results(
        gt_object=expected_output,
        projections_noisy=projections_noisy,
        recon=agent_recon,
        shape=shape,
        save_name="agent_vs_std"
    )

    print(f"\nComparison Metrics (Agent vs Standard):")
    print(f"PSNR: {p_agent:.2f} (Should be very high, e.g., > 40dB if identical)")
    print(f"SSIM: {s_agent:.4f} (Should be close to 1.0)")

    # Criteria: Since this is an optimization algorithm, floating point differences 
    # might occur. However, if the logic is identical, it should be very close.
    # If PSNR < 30 or SSIM < 0.95, something is likely wrong with the logic porting.
    
    if p_agent < 30.0 or s_agent < 0.95:
        print("\nFAILURE: Agent performance deviates significantly from Standard output.")
        sys.exit(1)
    else:
        print("\nSUCCESS: Agent result matches Standard result within acceptable margins.")
        sys.exit(0)

if __name__ == "__main__":
    run_test()