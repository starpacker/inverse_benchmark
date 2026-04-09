import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the target function
from agent_run_inversion import run_inversion


# --- Injected Referee (Evaluation Logic) ---

def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    """Interpolate element values to node values for visualization"""
    n_nodes = pts.shape[0]
    node_val = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    
    for i, el in enumerate(sim):
        node_val[el] += sim_values[i]
        count[el] += 1
        
    return node_val / np.maximum(count, 1)


def evaluate_results(
    ds: np.ndarray,
    pts: np.ndarray,
    tri: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    output_filename: str = "3D_eit.png"
) -> dict:
    """
    Evaluate and visualize the reconstruction results.
    
    Parameters
    ----------
    ds : np.ndarray
        Reconstructed conductivity change (element-wise)
    pts : np.ndarray
        Node coordinates
    tri : np.ndarray
        Element connectivity
    v0 : np.ndarray
        Baseline measurements
    v1 : np.ndarray
        Perturbed measurements
    output_filename : str
        Output filename for the visualization
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    print("Evaluating results...")

    node_ds = sim2pts(pts, tri, np.real(ds))

    ds_real = np.real(ds)
    ds_min = np.min(ds_real)
    ds_max = np.max(ds_real)
    ds_mean = np.mean(ds_real)
    ds_std = np.std(ds_real)

    dv = v1 - v0
    data_residual = np.linalg.norm(dv)
    relative_change = np.linalg.norm(dv) / np.linalg.norm(v0) if np.linalg.norm(v0) > 0 else 0.0

    max_change_element = np.argmax(np.abs(ds_real))
    max_change_value = ds_real[max_change_element]

    print(f"Reconstruction statistics:")
    print(f"  Min conductivity change: {ds_min:.6f}")
    print(f"  Max conductivity change: {ds_max:.6f}")
    print(f"  Mean conductivity change: {ds_mean:.6f}")
    print(f"  Std conductivity change: {ds_std:.6f}")
    print(f"  Data residual norm: {data_residual:.6f}")
    print(f"  Relative voltage change: {relative_change:.6f}")
    print(f"  Element with max change: {max_change_element} (value: {max_change_value:.6f})")

    print("Visualizing...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    im = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_ds, cmap='viridis', s=50, alpha=0.8)
    fig.colorbar(im, ax=ax, label='Conductivity Change')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D EIT Reconstruction (Refactored)')

    plt.savefig(output_filename)
    print(f"Saved visualization to {output_filename}")

    metrics = {
        'ds_min': ds_min,
        'ds_max': ds_max,
        'ds_mean': ds_mean,
        'ds_std': ds_std,
        'data_residual': data_residual,
        'relative_change': relative_change,
        'max_change_element': max_change_element,
        'max_change_value': max_change_value,
        'node_ds': node_ds
    }

    return metrics


# --- Main Test Logic ---

def main():
    data_paths = ['/home/yjh/pyeit_sandbox/examples/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_data_path = outer_data_files[0]
        print(f"Loading outer data from: {outer_data_path}")
        
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Loaded args: {len(args)} positional arguments")
        print(f"Loaded kwargs: {list(kwargs.keys())}")
        
        # Run the agent function
        print("\n--- Running Agent Function ---")
        agent_output = run_inversion(*args, **kwargs)
        print(f"Agent output shape: {agent_output.shape if hasattr(agent_output, 'shape') else type(agent_output)}")
        
        if is_chained:
            # Chained execution pattern
            print("\n--- Chained Execution Pattern Detected ---")
            inner_data_path = inner_data_files[0]
            print(f"Loading inner data from: {inner_data_path}")
            
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            print("\n--- Direct Execution Pattern ---")
            final_result = agent_output
            std_result = std_output
        
        # Extract mesh information for evaluation
        # The mesh_obj should be in the args or kwargs
        mesh_obj = None
        v0 = None
        v1 = None
        
        # Try to find mesh_obj from kwargs first, then args
        if 'mesh_obj' in kwargs:
            mesh_obj = kwargs['mesh_obj']
        elif len(args) > 2:
            mesh_obj = args[2]  # Based on function signature: v1, v0, mesh_obj, ...
        
        # Get v0 and v1
        if 'v0' in kwargs:
            v0 = kwargs['v0']
        elif len(args) > 1:
            v0 = args[1]
        
        if 'v1' in kwargs:
            v1 = kwargs['v1']
        elif len(args) > 0:
            v1 = args[0]
        
        if mesh_obj is None:
            print("ERROR: Could not find mesh_obj in inputs!")
            sys.exit(1)
        
        pts = mesh_obj.node
        tri = mesh_obj.element
        
        print(f"\nMesh info: {pts.shape[0]} nodes, {tri.shape[0]} elements")
        print(f"v0 shape: {v0.shape if hasattr(v0, 'shape') else type(v0)}")
        print(f"v1 shape: {v1.shape if hasattr(v1, 'shape') else type(v1)}")
        
        # Evaluate agent result
        print("\n--- Evaluating Agent Result ---")
        metrics_agent = evaluate_results(
            final_result, pts, tri, v0, v1, 
            output_filename="3D_eit_agent.png"
        )
        
        # Evaluate standard result
        print("\n--- Evaluating Standard Result ---")
        metrics_std = evaluate_results(
            std_result, pts, tri, v0, v1,
            output_filename="3D_eit_standard.png"
        )
        
        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        # Primary metrics for comparison
        score_agent = metrics_agent['ds_std']  # Using std as a measure of reconstruction quality
        score_std = metrics_std['ds_std']
        
        print(f"\nAgent ds_std: {score_agent:.6f}")
        print(f"Standard ds_std: {score_std:.6f}")
        
        # Also compare other key metrics
        print(f"\nAgent ds_min: {metrics_agent['ds_min']:.6f}, Standard ds_min: {metrics_std['ds_min']:.6f}")
        print(f"Agent ds_max: {metrics_agent['ds_max']:.6f}, Standard ds_max: {metrics_std['ds_max']:.6f}")
        print(f"Agent ds_mean: {metrics_agent['ds_mean']:.6f}, Standard ds_mean: {metrics_std['ds_mean']:.6f}")
        print(f"Agent max_change_element: {metrics_agent['max_change_element']}, Standard: {metrics_std['max_change_element']}")
        print(f"Agent max_change_value: {metrics_agent['max_change_value']:.6f}, Standard: {metrics_std['max_change_value']:.6f}")
        
        # Compute correlation between reconstructions
        correlation = np.corrcoef(np.real(final_result).flatten(), np.real(std_result).flatten())[0, 1]
        print(f"\nCorrelation between agent and standard reconstruction: {correlation:.6f}")
        
        # Compute relative error
        relative_error = np.linalg.norm(final_result - std_result) / np.linalg.norm(std_result)
        print(f"Relative error: {relative_error:.6f}")
        
        # Determine success
        # For reconstruction, we want high correlation and low relative error
        success = True
        tolerance = 0.1  # 10% tolerance
        
        # Check correlation (should be close to 1)
        if correlation < 0.9:
            print(f"\nWARNING: Correlation {correlation:.4f} is below threshold 0.9")
            success = False
        
        # Check relative error (should be small)
        if relative_error > tolerance:
            print(f"\nWARNING: Relative error {relative_error:.4f} exceeds tolerance {tolerance}")
            success = False
        
        # Check if key metrics are within tolerance
        if abs(metrics_agent['ds_max'] - metrics_std['ds_max']) / abs(metrics_std['ds_max']) > tolerance:
            print(f"\nWARNING: ds_max differs significantly")
            success = False
        
        if abs(metrics_agent['ds_min'] - metrics_std['ds_min']) / abs(metrics_std['ds_min']) > tolerance:
            print(f"\nWARNING: ds_min differs significantly")
            success = False
        
        print("\n" + "="*60)
        if success:
            print("TEST PASSED: Agent performance is acceptable")
            sys.exit(0)
        else:
            print("TEST FAILED: Agent performance degraded significantly")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()