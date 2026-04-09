import sys
import os
import dill
import numpy as np
import traceback
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Dependencies ---

def forward_operator(DM, angle, valid_cap, mode='trace'):
    """
    Forward operator that performs ray tracing or rendering.
    
    Parameters:
    -----------
    DM : DiffMetrology object
        The metrology object containing scene and lensgroup
    angle : float
        Rotation angle for the measurement
    valid_cap : torch.Tensor
        Valid pixel mask
    mode : str
        'trace' for ray tracing (returns intersection points)
        'render' for image rendering (returns rendered images)
    
    Returns:
    --------
    result : torch.Tensor
        Either intersection points (mode='trace') or rendered images (mode='render')
    """
    if mode == 'trace':
        # Ray tracing forward model - computes intersection points on display
        ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
        return ps
    elif mode == 'render':
        # Image rendering forward model - renders images from current parameters
        I = valid_cap * torch.stack(DM.render(with_element=True, angles=angle))
        I[torch.isnan(I)] = 0.0
        return I
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trace' or 'render'.")


def evaluate_results(DM, ps_cap, valid_cap, angle, I0, save_string="result"):
    """
    Evaluate and visualize the results of the inversion.
    
    Parameters:
    -----------
    DM : DiffMetrology object
        The metrology object with current parameters
    ps_cap : torch.Tensor
        Captured/measured intersection points
    valid_cap : torch.Tensor
        Valid pixel mask
    angle : float
        Rotation angle
    I0 : torch.Tensor
        Reference images for comparison
    save_string : str
        Prefix for saved figure filenames
    
    Returns:
    --------
    error : float
        Mean displacement error in mm
    """
    print("Showing lens parameters:")
    for i in range(len(DM.scene.lensgroup.surfaces)):
        c_val = DM.scene.lensgroup.surfaces[i].c
        if hasattr(c_val, 'detach'):
            c_val = c_val.detach()
        if hasattr(c_val, 'item'):
            c_val = c_val.item()
        print(f"Lens radius of curvature at surface[{i}]: {1.0 / c_val}")
    
    d_val = DM.scene.lensgroup.surfaces[1].d
    if hasattr(d_val, 'detach'):
        d_val = d_val.detach()
    print(d_val)

    print("Visualize status.")
    ps_current = forward_operator(DM, angle, valid_cap, mode='trace')

    print("Showing spot diagrams at display.")
    # Detach tensors for spot diagram
    ps_cap_np = ps_cap.detach() if hasattr(ps_cap, 'detach') else ps_cap
    ps_current_np = ps_current.detach() if hasattr(ps_current, 'detach') else ps_current
    valid_cap_np = valid_cap.detach() if hasattr(valid_cap, 'detach') else valid_cap
    
    try:
        DM.spot_diagram(ps_cap_np, ps_current_np, valid=valid_cap_np, angle=angle, with_grid=False)
        plt.savefig(f"{save_string}_spot_diagram.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate spot diagram: {e}")
        plt.close('all')

    print("Showing images (measurement & modeled & |measurement - modeled|).")

    # Render images from parameters
    I = forward_operator(DM, angle, valid_cap, mode='render')

    # Detach tensors for plotting
    I_detached = I.detach() if hasattr(I, 'detach') else I
    I0_detached = I0.detach() if hasattr(I0, 'detach') else I0

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i in range(2):
            im = axes[i, 0].imshow(I0_detached[i].cpu().numpy(), vmin=0, vmax=1, cmap='gray')
            axes[i, 0].set_title(f"Camera {i + 1}\nMeasurement")
            axes[i, 0].set_xlabel('[pixel]')
            axes[i, 0].set_ylabel('[pixel]')
            plt.colorbar(im, ax=axes[i, 0])

            im = axes[i, 1].imshow(I_detached[i].cpu().numpy(), vmin=0, vmax=1, cmap='gray')
            plt.colorbar(im, ax=axes[i, 1])
            axes[i, 1].set_title(f"Camera {i + 1}\nModeled")
            axes[i, 1].set_xlabel('[pixel]')
            axes[i, 1].set_ylabel('[pixel]')

            im = axes[i, 2].imshow(I0_detached[i].cpu().numpy() - I_detached[i].cpu().numpy(), vmin=-1, vmax=1, cmap='coolwarm')
            plt.colorbar(im, ax=axes[i, 2])
            axes[i, 2].set_title(f"Camera {i + 1}\nError")
            axes[i, 2].set_xlabel('[pixel]')
            axes[i, 2].set_ylabel('[pixel]')

        fig.suptitle(save_string)
        fig.savefig(f"{save_string}_images.jpg", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate images plot: {e}")
        plt.close('all')

    # Print mean displacement error - ensure detached
    ps_current_detached = ps_current.detach() if hasattr(ps_current, 'detach') else ps_current
    ps_cap_detached = ps_cap.detach() if hasattr(ps_cap, 'detach') else ps_cap
    valid_cap_detached = valid_cap.detach() if hasattr(valid_cap, 'detach') else valid_cap
    
    T = ps_current_detached - ps_cap_detached
    E = torch.sqrt(torch.sum(T[valid_cap_detached, ...] ** 2, axis=-1)).mean()
    print("error = {} [um]".format(E.item() * 1e3))

    return E.item()


def main():
    data_paths = ['/home/yjh/diff_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        if 'parent_function' in os.path.basename(path):
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer data files: {outer_files}")
    print(f"Inner data files: {inner_files}")
    
    # Load outer data
    if not outer_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"Loading outer data from: {outer_path}")
    
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args count: {len(args)}")
    print(f"Kwargs keys: {kwargs.keys()}")
    
    # Run agent's function
    print("Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent function: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Agent output type: {type(agent_output)}")
    print(f"Standard output type: {type(std_output)}")
    
    # Extract DM and loss history from outputs
    # run_inversion returns (ls, DM) where ls is loss history
    agent_ls, agent_DM = agent_output
    std_ls, std_DM = std_output
    
    # Convert loss histories to numpy for comparison
    agent_ls_np = np.array([l.detach().cpu().numpy() if hasattr(l, 'detach') else l for l in agent_ls])
    std_ls_np = np.array([l.detach().cpu().numpy() if hasattr(l, 'detach') else l for l in std_ls])
    
    print(f"Agent loss history: {agent_ls_np}")
    print(f"Standard loss history: {std_ls_np}")
    
    # Extract inputs needed for evaluation
    # args = (DM, ps_cap, valid_cap, angle, device)
    DM_input, ps_cap, valid_cap, angle, device = args[:5]
    
    # Try to get I0 for visualization - generate it if not available
    if 'I0' in outer_data:
        I0 = outer_data['I0']
    else:
        print("I0 not found in data, attempting to generate from standard DM...")
        try:
            # Generate reference images from standard DM
            I0 = forward_operator(std_DM, angle, valid_cap, mode='render')
        except Exception as e:
            print(f"Warning: Could not generate I0: {e}")
            # Create dummy I0
            I0 = torch.zeros(2, valid_cap.shape[0], valid_cap.shape[1], device=device)
    
    # Evaluate agent results
    print("\n=== Evaluating Agent Results ===")
    try:
        score_agent = evaluate_results(agent_DM, ps_cap, valid_cap, angle, I0, save_string="agent_result")
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        # Fall back to using final loss as metric
        score_agent = float(agent_ls_np[-1])
        print(f"Using final loss as agent score: {score_agent}")
    
    # Evaluate standard results
    print("\n=== Evaluating Standard Results ===")
    try:
        score_std = evaluate_results(std_DM, ps_cap, valid_cap, angle, I0, save_string="std_result")
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        # Fall back to using final loss as metric
        score_std = float(std_ls_np[-1])
        print(f"Using final loss as standard score: {score_std}")
    
    print(f"\nScores -> Agent: {score_agent}, Standard: {score_std}")
    
    # For error metric, lower is better
    # Allow 10% margin of error
    margin = 0.10
    
    # Compare final losses as well
    agent_final_loss = float(agent_ls_np[-1])
    std_final_loss = float(std_ls_np[-1])
    print(f"Final Losses -> Agent: {agent_final_loss}, Standard: {std_final_loss}")
    
    # Check if agent performance is acceptable
    # For error/loss metrics, lower is better
    # Agent should not be significantly worse than standard
    
    loss_ratio = agent_final_loss / std_final_loss if std_final_loss != 0 else float('inf')
    error_ratio = score_agent / score_std if score_std != 0 else float('inf')
    
    print(f"Loss ratio (agent/std): {loss_ratio}")
    print(f"Error ratio (agent/std): {error_ratio}")
    
    # Allow up to 10% worse performance
    threshold = 1.0 + margin
    
    if loss_ratio <= threshold and error_ratio <= threshold:
        print(f"\nSUCCESS: Agent performance is acceptable (within {margin*100}% margin)")
        print(f"Loss ratio: {loss_ratio:.4f} <= {threshold}")
        print(f"Error ratio: {error_ratio:.4f} <= {threshold}")
        sys.exit(0)
    else:
        print(f"\nFAILURE: Agent performance degraded significantly")
        if loss_ratio > threshold:
            print(f"Loss ratio: {loss_ratio:.4f} > {threshold}")
        if error_ratio > threshold:
            print(f"Error ratio: {error_ratio:.4f} > {threshold}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)