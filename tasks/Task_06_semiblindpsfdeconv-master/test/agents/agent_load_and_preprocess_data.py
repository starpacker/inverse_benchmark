import os

import logging

import numpy as np

from skimage import io, metrics, util

import torch

import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

log = logging.getLogger('semiblind')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(img_path, crop_size=512, model_path=None):
    """
    Loads image, preprocesses it (crop/gray), and loads the neural network model.
    """
    # Load Image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    original_img = io.imread(img_path).astype(np.float32) / 255.0
    
    if len(original_img.shape) == 3:
        original_img = np.mean(original_img, axis=2)
        
    h, w = original_img.shape
    cx, cy = h//2, w//2
    start_x = max(0, cx - crop_size//2)
    start_y = max(0, cy - crop_size//2)
    end_x = min(h, start_x + crop_size)
    end_y = min(w, start_y + crop_size)
    
    gt_img = original_img[start_x:end_x, start_y:end_y]
    
    # Ensure divisible by 64
    h_new = (gt_img.shape[0] // 64) * 64
    w_new = (gt_img.shape[1] // 64) * 64
    gt_img = gt_img[:h_new, :w_new]
    
    # Load Model
    cnn_model = None
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        try:
            cnn_model = torch.load(model_path, map_location=device, weights_only=False)
            for m in cnn_model.modules():
                if isinstance(m, nn.AvgPool2d):
                    if not hasattr(m, 'divisor_override'):
                        m.divisor_override = None
            cnn_model.to(device)
            cnn_model.eval()
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise
            
    return gt_img, cnn_model
