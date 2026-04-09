import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def unpad(img, npad):
    return img[npad:-npad, npad:-npad]
