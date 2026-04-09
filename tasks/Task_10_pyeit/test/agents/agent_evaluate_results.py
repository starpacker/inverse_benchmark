import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray) -> np.ndarray:
    """
    Interpolate element values to node values for visualization.
    
    Args:
        pts: (N_nodes, 3) array of node coordinates.
        sim: (N_elems, 4) array of element connectivity (indices of nodes).
        sim_values: (N_elems,) array of values defined on elements (e.g., conductivity).
        
    Returns:
        node_val: (N_nodes,) array of interpolated values at each node.
    """
    n_nodes = pts.shape[0]
    node_val = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    
    # Iterate over every element to accumulate values at its constituent nodes
    for i, el in enumerate(sim):
        node_val[el] += sim_values[i]
        count[el] += 1
        
    # Average the accumulated values. 
    # np.maximum ensures we do not divide by zero for isolated nodes.
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
        Reconstructed conductivity change (element-wise).
    pts : np.ndarray
        Node coordinates (N, 3).
    tri : np.ndarray
        Element connectivity (M, 4) for tetrahedrons.
    v0 : np.ndarray
        Baseline measurements (boundary voltages).
    v1 : np.ndarray
        Perturbed measurements (boundary voltages).
    output_filename : str
        Output filename for the visualization.
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    print("Evaluating results...")

    # 1. Interpolate element-wise conductivity to node-wise for plotting
    node_ds = sim2pts(pts, tri, np.real(ds))

    # 2. Calculate Image-Space Statistics
    ds_real = np.real(ds)
    ds_min = np.min(ds_real)
    ds_max = np.max(ds_real)
    ds_mean = np.mean(ds_real)
    ds_std = np.std(ds_real)

    # 3. Calculate Data-Space Residuals
    dv = v1 - v0
    data_residual = np.linalg.norm(dv)
    # Avoid division by zero if v0 is empty or zero-vector
    relative_change = np.linalg.norm(dv) / np.linalg.norm(v0) if np.linalg.norm(v0) > 0 else 0.0

    # 4. Identify Peak Anomaly
    max_change_element = np.argmax(np.abs(ds_real))
    max_change_value = ds_real[max_change_element]

    # 5. Console Reporting
    print(f"Reconstruction statistics:")
    print(f"  Min conductivity change: {ds_min:.6f}")
    print(f"  Max conductivity change: {ds_max:.6f}")
    print(f"  Mean conductivity change: {ds_mean:.6f}")
    print(f"  Std conductivity change: {ds_std:.6f}")
    print(f"  Data residual norm: {data_residual:.6f}")
    print(f"  Relative voltage change: {relative_change:.6f}")
    print(f"  Element with max change: {max_change_element} (value: {max_change_value:.6f})")

    # 6. Visualization
    print("Visualizing...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: x, y, z coordinates colored by node_ds magnitude
    im = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_ds, cmap='viridis', s=50, alpha=0.8)
    fig.colorbar(im, ax=ax, label='Conductivity Change')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D EIT Reconstruction (Refactored)')

    plt.savefig(output_filename)
    print(f"Saved visualization to {output_filename}")

    # 7. Return Metrics Dictionary
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