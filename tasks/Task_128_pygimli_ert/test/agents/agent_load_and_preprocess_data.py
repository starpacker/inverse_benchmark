import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

import pygimli as pg

import pygimli.meshtools as mt

from pygimli.physics import ert

def load_and_preprocess_data(results_dir):
    """
    Create synthetic subsurface model, electrode array, and measurement scheme.
    
    Returns
    -------
    dict containing:
        - mesh: forward mesh
        - scheme: measurement scheme
        - geom: geometry PLC
        - rhomap: resistivity map
        - gt_res_np: ground truth resistivity array
        - results_dir: output directory path
    """
    print("=" * 60)
    print("pyGIMLi ERT Inversion Benchmark")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)

    print("\n[1/6] Creating synthetic subsurface model...")

    # Create the world geometry: 80m wide, 30m deep
    world = mt.createWorld(start=[-40, 0], end=[40, -30],
                           layers=[-5, -15], worldMarker=True)

    # Add resistivity anomalies
    block1 = mt.createRectangle(start=[-15, -3], end=[-5, -10],
                                marker=4, area=0.3)
    block2 = mt.createRectangle(start=[5, -5], end=[15, -12],
                                marker=5, area=0.3)
    circle = mt.createCircle(pos=[0, -18], radius=3, marker=6, area=0.3)

    geom = world + block1 + block2 + circle

    print("[2/6] Setting up electrode array (Dipole-dipole, 41 electrodes)...")

    scheme = ert.createData(elecs=np.linspace(start=-20, stop=20, num=41),
                            schemeName='dd')

    print(f"   Number of electrodes: {scheme.sensorCount()}")
    print(f"   Number of measurements: {scheme.size()}")

    for p in scheme.sensors():
        geom.createNode(p)
        geom.createNode(p - [0, 0.1])

    mesh = mt.createMesh(geom, quality=34)
    print(f"   Forward mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")

    # Define resistivity distribution (ground truth)
    rhomap = [
        [0, 100.0],
        [1, 100.0],
        [2, 50.0],
        [3, 200.0],
        [4, 10.0],
        [5, 500.0],
        [6, 5.0],
    ]

    gt_res = pg.solver.parseArgToArray(rhomap, mesh.cellCount(), mesh)
    gt_res_np = np.array(gt_res)
    print(f"   GT resistivity range: [{gt_res_np.min():.1f}, {gt_res_np.max():.1f}] Ohm·m")

    return {
        'mesh': mesh,
        'scheme': scheme,
        'geom': geom,
        'rhomap': rhomap,
        'gt_res_np': gt_res_np,
        'results_dir': results_dir
    }
