import logging

import numpy as np

from scipy import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def envel_detect(scan_line):
    """Envelope detection using Hilbert transform"""
    envelope = np.abs(signal.hilbert(scan_line))
    return envelope
