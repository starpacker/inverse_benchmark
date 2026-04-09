import os

import numpy as np

import tifffile

import pooch

def fetch(fname):
    """Fetches data from remote or local source."""
    if os.path.exists(fname):
        return os.path.abspath(fname)
        
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    file_path = os.path.join(script_dir, fname)
    if os.path.exists(file_path):
        return file_path

    if fname == 'Embryo.tif':
        url = 'https://github.com/phasorpy/phasorpy-data/raw/main/zenodo_8046636/Embryo.tif'
        file_hash = 'd1107de8d0f3da476e90bcb80ddf40231df343ed9f28340c873cf858ca869e20'
        return pooch.retrieve(
            url=url,
            known_hash='sha256:' + file_hash,
            fname=fname,
            path=pooch.os_cache('phasorpy'),
            progressbar=True
        )
    return fname

def load_and_preprocess_data(filename, harmonic='all'):
    """
    Loads image data and determines dimensions.
    Returns the raw data array and preprocessing metadata.
    """
    local_path = fetch(filename)
    
    # Load using tifffile
    with tifffile.TiffFile(local_path) as tif:
        data = tif.asarray()
        
        # Heuristic for Embryo.tif or general timelapse FLIM data
        # Embryo.tif is typically (Time, Y, X) = (56, 128, 128)
        shape = data.shape
        dims = list('CYX') # Default guess
        
        if len(shape) == 3:
            # Assume (Time/Harmonic, Y, X)
            dims = ['H', 'Y', 'X'] 
        elif len(shape) == 4:
            dims = ['C', 'H', 'Y', 'X']
            
        # Basic normalization or type conversion if needed
        data = data.astype(np.float32)

    return data, {'dims': dims, 'shape': shape, 'source': local_path}
