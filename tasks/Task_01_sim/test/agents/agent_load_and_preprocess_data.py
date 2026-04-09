import numpy as np
import pywt
from skimage import io

def Low_frequency_resolve(coeffs, dlevel):
    cA = coeffs[0]
    vec = [cA] + [(np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)) for cH, cV, cD in coeffs[1:]]
    return vec

def rm_1(Biter, x, y):
    return Biter[:x, :y]

def background_estimation(imgs, th=1, dlevel=7, wavename='db6', iter=3):
    if imgs.ndim == 3:
        stack = True
    else:
        stack = False
        imgs = imgs[np.newaxis, ...]

    backgrounds = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        res = imgs[i]
        for _ in range(iter):
            coeffs = pywt.wavedec2(res, wavename, level=dlevel)
            low_freq_coeffs = Low_frequency_resolve(coeffs, dlevel)
            Biter = pywt.waverec2(low_freq_coeffs, wavename)
            Biter = rm_1(Biter, *res.shape)
            if th > 0:
                epsilon = np.sqrt(np.abs(res)) / 2
                mask = res < (Biter + epsilon)
                res[mask] = Biter[mask]
        backgrounds[i] = Biter

    return backgrounds if stack else backgrounds[0]

def spatial_upsample(SIMmovie, n=2):
    y = np.zeros((SIMmovie.shape[0] * n, SIMmovie.shape[1] * n), dtype=SIMmovie.dtype)
    y[0::n, 0::n] = SIMmovie
    return y

def fInterp_2D(img, newsz):
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    center = tuple(map(lambda x: x // 2, F_shifted.shape))
    img_ip = np.zeros(newsz, dtype=complex)
    img_ip[:center[0], :center[1]] = F_shifted[:center[0], :center[1]]
    img_ip[-center[0]:, :center[1]] = F_shifted[-center[0]:, :center[1]]
    img_ip[:center[0], -center[1]:] = F_shifted[:center[0], -center[1]:]
    img_ip[-center[0]:, -center[1]:] = F_shifted[-center[0]:, -center[1]:]
    img_ip_shifted = np.fft.ifftshift(img_ip)
    img_up = np.fft.ifft2(img_ip_shifted)
    return np.abs(img_up) * (newsz[0] * newsz[1]) / (img.shape[0] * img.shape[1])

def fourier_upsample(imgstack, n=2):
    upsampled_stack = []
    for img in imgstack:
        padded_img = np.pad(img, ((img.shape[0]//2, img.shape[0]//2), (img.shape[1]//2, img.shape[1]//2)), mode='symmetric')
        target_shape = (padded_img.shape[0] * n, padded_img.shape[1] * n)
        upsampled_img = fInterp_2D(padded_img, target_shape)
        crop_start = (upsampled_img.shape[0]//4, upsampled_img.shape[1]//4)
        crop_end = (crop_start[0] + img.shape[0] * n, crop_start[1] + img.shape[1] * n)
        upsampled_stack.append(upsampled_img[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
    return np.array(upsampled_stack)

def load_and_preprocess_data(input_path, up_sample=0):
    img = io.imread(input_path).astype(float)
    scaler = img.max()
    img /= scaler
    backgrounds = background_estimation(img / 2.5)
    img = np.clip(img - backgrounds, 0, None)
    if up_sample == 1:
        img = fourier_upsample(img[np.newaxis, ...])[0]
    elif up_sample == 2:
        img = spatial_upsample(img)
    img = np.clip(img, 0, 1)
    return img, scaler