from lensless.utils.io import load_data

def load_and_preprocess_data(psf_path, data_path, downsample=4):
    """
    Load PSF and measurement data, preprocess them for reconstruction.
    
    Parameters
    ----------
    psf_path : str
        Path to the PSF image file.
    data_path : str
        Path to the measurement/raw data image file.
    downsample : int
        Downsampling factor for loading data.
    
    Returns
    -------
    dict
        Dictionary containing 'psf' and 'data' numpy arrays.
    """
    print(f"Loading data from {data_path}...")
    print(f"Loading PSF from {psf_path}...")
    
    # The load_data utility handles reading, converting to float32,
    # downsampling, and normalizing the images to [0, 1].
    psf, data = load_data(
        psf_fp=psf_path,
        data_fp=data_path,
        background_fp=None,
        dtype="float32",
        downsample=downsample,
        bayer=False,
        plot=False,
        flip=False,
        normalize=True
    )
    
    print(f"Data shape: {data.shape}")
    print(f"PSF shape: {psf.shape}")
    
    return {"psf": psf, "data": data}