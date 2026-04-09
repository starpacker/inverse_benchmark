import os

import numpy as np

import xml.etree.ElementTree as ET

def load_and_preprocess_data(directory, frame_idx=0):
    """
    Loads metadata, calibration files (chirp, dispersion), and raw OCT data for a specific frame.
    Prepares reconstruction parameters.
    """
    # Initialize container for paths and settings
    filenames = {}
    settings = {}
    
    # 1. Locate files
    for f in os.listdir(directory):
        if f.endswith('.ofd'):
            filenames['ofd'] = os.path.join(directory, f)
            basename = f[:-4]
            break
    
    if 'ofd' not in filenames:
        raise FileNotFoundError("No .ofd file found in directory")

    # Locate XML
    settings_dir = os.path.join(directory, 'settings')
    if os.path.exists(settings_dir):
        for f in os.listdir(settings_dir):
            if f.endswith('_info.xml') and basename in f:
                filenames['xml'] = os.path.join(settings_dir, f)
                break
    
    if 'xml' not in filenames:
        for f in os.listdir(directory):
            if f.endswith('_info.xml'):
                filenames['xml'] = os.path.join(directory, f)
                break
                
    if 'xml' not in filenames:
         raise FileNotFoundError("No info XML file found.")

    # Locate Calibration
    for f in os.listdir(directory):
        if f.endswith('.laser'):
            filenames['chirp'] = os.path.join(directory, f)
        if f.endswith('.dispersion'):
            filenames['dispersion'] = os.path.join(directory, f)

    # 2. Parse XML
    tree = ET.parse(filenames['xml'])
    root = tree.getroot()
    xml_info = {child.tag: child.text for child in root.iter()}

    settings['numSamples'] = int(xml_info.get('totalSamplesPerALinePerChannel', 2048))
    # Note: Using hardcoded value based on context provided in prompt logic usually
    # But sticking to XML or default. 
    settings['numAlines'] = int(xml_info.get('totalALinesPerProcessedBScan', 1024))
    settings['zoomFactor'] = int(xml_info.get('proc_zoomlevel', 1))
    
    if 'postproc_zscans' in xml_info:
        settings['numZOut'] = int(xml_info['postproc_zscans'])
    elif 'proc_nzscans' in xml_info:
         settings['numZOut'] = int(xml_info['proc_nzscans'])
    else:
         settings['numZOut'] = 1024

    demod_str = xml_info.get('proc_demodulation', '0.5,0,1,0,0,0')
    settings['demodSet'] = [float(x.strip()) for x in demod_str.split(',')]
    settings['clockRateMHz'] = int(xml_info.get('captureClockRateMHz', 100))
    settings['flipUpDown'] = (xml_info.get('proc_flipimagealines', 'False').lower() in ("yes", "true", "t", "1"))
    
    loglim_str = xml_info.get('proc_intensityloglim', '-40,130')
    settings['contrastLowHigh'] = [float(x) for x in loglim_str.split(',')]
    settings['invertGray'] = (xml_info.get('proc_intensityinvertgrayscale', 'False').lower() in ("yes", "true", "t", "1"))

    # Frame Size Calculation (Interlaced 2-channel, uint16)
    # Common standard: 3072 A-lines per raw frame for this system type
    settings['numAlinesPerRawFrame'] = 3072 
    settings['samplesPerFrame'] = settings['numSamples'] * settings['numAlinesPerRawFrame'] * 2
    settings['bytesPerFrame'] = settings['samplesPerFrame'] * 2

    # 3. Load Calibration Data
    chirp_raw = None
    if 'chirp' in filenames:
        chirp_raw = np.fromfile(filenames['chirp'], dtype='float32')[1:] # Skip header
    
    dispersion_raw = None
    if 'dispersion' in filenames:
        dispersion_raw = np.fromfile(filenames['dispersion'], dtype='float32')[1:]

    # 4. Precompute Reconstruction Maps
    # A. Chirp Map
    num_samples = settings['numSamples']
    fourier_length = num_samples // 2
    zoom = settings['zoomFactor']
    
    if chirp_raw is not None:
        chirp_len = len(chirp_raw)
        xchirp = np.linspace(0, 1, chirp_len)
        xsample = np.linspace(0, 1, fourier_length)
        interp_chirp = np.interp(xsample, xchirp, chirp_raw)
        scaled_chirp = interp_chirp * num_samples * zoom
        # Boundary checks
        scaled_chirp[0] = 1
        scaled_chirp[-1] = num_samples * zoom - 1
        chirp_indices = np.floor(scaled_chirp).astype(int)
        chirp_indices = np.clip(chirp_indices, 0, num_samples * zoom - 1)
    else:
        chirp_indices = np.arange(fourier_length) * zoom * 2

    settings['chirp_indices'] = chirp_indices

    # B. Dispersion Vector
    if dispersion_raw is not None:
        d_len = len(dispersion_raw) // 2
        disp_complex = dispersion_raw[:d_len] + 1j * dispersion_raw[d_len:]
        xdisp = np.linspace(0, 1, d_len)
        xsample = np.linspace(0, 1, fourier_length)
        real_interp = np.interp(xsample, xdisp, disp_complex.real)
        imag_interp = np.interp(xsample, xdisp, disp_complex.imag)
        dispersion_vector = real_interp + 1j * imag_interp
    else:
        dispersion_vector = np.ones(fourier_length, dtype='complex64')
    
    settings['dispersion_vector'] = dispersion_vector

    # C. Demodulation Indices
    clock_rate = settings['clockRateMHz']
    demod_val = settings['demodSet'][0]
    carrier = clock_rate * demod_val / 2
    norm_carrier = carrier / (clock_rate / 2)
    
    demod_idx = int(round(norm_carrier * 0.5 * num_samples))
    demod_rev_idx = int(round(demod_idx * settings['numZOut'] / num_samples * 2))
    
    settings['demod_idx'] = demod_idx
    settings['demod_rev_idx'] = demod_rev_idx
    
    # D. Window
    settings['fringe_window'] = np.hanning(fourier_length)[:, None]

    # 5. Load Raw Frame
    offset = frame_idx * settings['bytesPerFrame']
    with open(filenames['ofd'], 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype='uint16', count=settings['samplesPerFrame'])
    
    # De-interlace and reshape
    ch1_raw = raw_data[0::2]
    ch2_raw = raw_data[1::2]
    
    raw_frame_ch1 = ch1_raw.reshape((settings['numSamples'], settings['numAlinesPerRawFrame']), order='F')
    raw_frame_ch2 = ch2_raw.reshape((settings['numSamples'], settings['numAlinesPerRawFrame']), order='F')

    return raw_frame_ch1, raw_frame_ch2, settings
