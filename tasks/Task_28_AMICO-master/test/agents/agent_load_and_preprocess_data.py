import os

import warnings

import numpy as np

import nibabel as nib

from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

warnings.filterwarnings("ignore")

class Scheme:
    def __init__(self, filename, b0_thr=0):
        self.version = None
        
        if isinstance(filename, str):
            if not os.path.isfile(filename):
                raise FileNotFoundError(f'Scheme file "{filename}" not found')
            try:
                self.raw = np.loadtxt(filename)
            except Exception as e:
                raise RuntimeError(f'Could not load scheme file: {e}')
            
            with open(filename, 'r') as f:
                first_line = f.readline()
                if 'VERSION: STEJSKALTANNER' in first_line:
                    self.version = 1
                else:
                    self.version = 0
        else:
            self.raw = filename
            self.version = 0 if self.raw.shape[1] <= 4 else 1

        self.nS = self.raw.shape[0]
        
        if self.version == 0:
            self.b = self.raw[:, 3]
            self.g = np.ones(self.nS)
        else:
            self.g = self.raw[:, 3]
            self.Delta = self.raw[:, 4]
            self.delta = self.raw[:, 5]
            self.TE = self.raw[:, 6]
            gamma = 2.675987e8
            self.b = (gamma * self.delta * self.g)**2 * (self.Delta - self.delta/3.0) * 1e-6 

        self.raw[:, :3] /= np.linalg.norm(self.raw[:, :3], axis=1)[:, None] + 1e-16
        
        self.b0_idx = np.where(self.b <= b0_thr)[0]
        self.dwi_idx = np.where(self.b > b0_thr)[0]
        self.b0_count = len(self.b0_idx)
        self.dwi_count = len(self.dwi_idx)
        
        b_rounded = np.round(self.b[self.dwi_idx], -2)
        unique_b = np.unique(b_rounded)
        self.shells = []
        for ub in unique_b:
            idx = self.dwi_idx[b_rounded == ub]
            shell = {'b': np.mean(self.b[idx]), 'idx': idx}
            if self.version == 1:
                shell['G'] = np.mean(self.g[idx])
                shell['Delta'] = np.mean(self.Delta[idx])
                shell['delta'] = np.mean(self.delta[idx])
                shell['TE'] = np.mean(self.TE[idx])
            shell['grad'] = self.raw[idx, :3]
            self.shells.append(shell)

def load_and_preprocess_data(dwi_file, scheme_file, mask_file):
    """
    Loads DWI data, scheme, mask, and computes Principal Directions (DTI).
    """
    print(f"\n[LOAD] Loading {dwi_file} and {scheme_file}...")
    
    img = nib.load(dwi_file)
    data = img.get_fdata()
    affine = img.affine
    
    scheme = Scheme(scheme_file, b0_thr=10)
    
    if mask_file and os.path.exists(mask_file):
        mask = nib.load(mask_file).get_fdata() > 0
    else:
        mask = np.ones(data.shape[:3], dtype=bool)
        
    # DTI Fit for Principal Directions
    print("[PREPROCESS] Estimating Principal Directions (DTI)...")
    gtab = gradient_table(scheme.b, scheme.raw[:, :3])
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    dirs = tenfit.evecs[..., 0] 
    
    return data, affine, mask, scheme, dirs
