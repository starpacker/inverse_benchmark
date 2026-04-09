import os

import numpy as np

import tifffile

import yaml

try:
    from pyolaf.geometry import LFM_computeGeometryParameters, LFM_setCameraParams_v2
    from pyolaf.lf import LFM_computeLFMatrixOperators
    from pyolaf.transform import LFM_retrieveTransformation, format_transform, get_transformed_shape, transform_img
    from pyolaf.project import LFM_forwardProject, LFM_backwardProject
    from pyolaf.aliasing import lanczosfft, LFM_computeDepthAdaptiveWidth
    HAS_PYOLAF = True
except ImportError:
    HAS_PYOLAF = False

def load_and_preprocess_data(data_path: str, new_spacing_px: int, depth_range: list, depth_step: float, super_res_factor: int):
    """
    Loads the required LFM data files and computes geometry parameters.

    Args:
        data_path (str): Path to the directory containing 'calib.tif', 'config.yaml', and the image file.
        new_spacing_px (int): New lenslet spacing in pixels.
        depth_range (list): [min_depth, max_depth] in mm.
        depth_step (float): Depth step in mm.
        super_res_factor (int): Super-resolution factor.

    Returns:
        dict: A dictionary containing all loaded and preprocessed data.
    """
    fname_calib = os.path.join(data_path, 'calib.tif')
    fname_config = os.path.join(data_path, 'config.yaml')
    fname_img = os.path.join(data_path, 'example_fly.tif')

    if not os.path.exists(fname_calib) or not os.path.exists(fname_config) or not os.path.exists(fname_img):
        raise FileNotFoundError(f"Required data files not found in {data_path}. Check README.")

    white_image = tifffile.imread(fname_calib).astype(np.float32)
    with open(fname_config, 'r') as f:
        config_dict = yaml.safe_load(f)
    raw_lenslet_image = tifffile.imread(fname_img).astype(np.float32)

    # Calculate camera parameters and geometry
    Camera = LFM_setCameraParams_v2(config_dict, new_spacing_px)
    LensletCenters, Resolution, LensletGridModel, NewLensletGridModel = \
        LFM_computeGeometryParameters(
            Camera, white_image, depth_range, depth_step, super_res_factor, False)
    
    # Compute forward and backward projection operators
    H, Ht = LFM_computeLFMatrixOperators(Camera, Resolution, LensletCenters)

    # Obtain transformation
    FixAll = LFM_retrieveTransformation(LensletGridModel, NewLensletGridModel)
    trans = format_transform(FixAll)
    imgSize = get_transformed_shape(white_image.shape, trans)
    imgSize = imgSize + (1 - np.remainder(imgSize, 2))  # Ensure even size

    texSize = np.ceil(np.multiply(imgSize, Resolution['texScaleFactor'])).astype('int32')
    texSize = texSize + (1 - np.remainder(texSize, 2))  # Ensure even size

    ndepths = len(Resolution['depths'])
    volumeSize = np.append(texSize, ndepths).astype('int32')

    return {
        'white_image': white_image,
        'config_dict': config_dict,
        'raw_lenslet_image': raw_lenslet_image,
        'Camera': Camera,
        'LensletCenters': LensletCenters,
        'Resolution': Resolution,
        'LensletGridModel': LensletGridModel,
        'NewLensletGridModel': NewLensletGridModel,
        'H': H,
        'Ht': Ht,
        'trans': trans,
        'imgSize': imgSize,
        'texSize': texSize,
        'volumeSize': volumeSize
    }
