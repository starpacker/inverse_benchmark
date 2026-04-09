import numpy as np

def custom_resize(img, factor):
    """
    Downsample image by a factor of 1/2^k using box filtering.
    """
    num = int(-np.log2(factor))
    for i in range(num):
        h, w = img.shape[:2]
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        img = img[:h_even, :w_even]
        img = 0.25 * (img[::2, ::2, ...] + img[1::2, ::2, ...] + img[::2, 1::2, ...] + img[1::2, 1::2, ...])
    return img
