import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
from scipy.fftpack import dct
import warnings

class StegoEvaluator:
    @staticmethod
    def calculate_mse(original, stego):
        diff = original.astype(np.float64) - stego.astype(np.float64)
        return np.mean(diff ** 2)

    @staticmethod
    def calculate_psnr(original, stego, max_pixel=255.0):
        mse = StegoEvaluator.calculate_mse(original, stego)
        if mse == 0:
            return 100.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return min(psnr, 100.0)

    @staticmethod
    def calculate_snr(original, stego):
        orig = original.astype(np.float64)
        steg = stego.astype(np.float64)
        signal_power = np.sum(orig ** 2)
        noise_power = np.sum((orig - steg) ** 2)
        if noise_power == 0:
            return 100.0
        snr = 10 * np.log10(signal_power / noise_power)
        return min(snr, 100.0)

    @staticmethod
    def calculate_ssim(original, stego):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ssim_val = ssim(original, stego, channel_axis=2, data_range=255)
            except:
                ssim_val = 1.0
        return float(ssim_val)

    @staticmethod
    def calculate_epi(img1, img2):
        g1 = np.mean(img1, axis=2)
        g2 = np.mean(img2, axis=2)
        s1 = sobel(g1)
        s2 = sobel(g2)
        s1_m = s1 - np.mean(s1)
        s2_m = s2 - np.mean(s2)
        num = np.sum(s1_m * s2_m)
        den = np.sqrt(np.sum(s1_m**2) * np.sum(s2_m**2))
        if den == 0:
            return 1.0 if num == 0 else 0.0
        epi = num / den
        return np.clip(epi, 0.0, 1.0)

    @staticmethod
    def calculate_entropy(image_arr):
        hist, _ = np.histogram(image_arr.flatten(), bins=256, range=(0,255))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    @staticmethod
    def embedding_capacity(payload_bytes, image_shape):
        total_pixels = image_shape[0] * image_shape[1]
        embedded_bits = len(payload_bytes) * 8
        return embedded_bits / total_pixels if total_pixels > 0 else 0.0

    @staticmethod
    def calculate_fdm(original, stego):
        def dct2d(block):
            return dct(dct(block.T, norm='ortho').T, norm='ortho')
        h, w, c = original.shape
        total_diff = 0.0
        for ch in range(c):
            orig_ch = original[:,:,ch].astype(np.float64)
            stego_ch = stego[:,:,ch].astype(np.float64)
            dct_orig = dct2d(orig_ch)
            dct_stego = dct2d(stego_ch)
            diff = np.mean((dct_orig - dct_stego) ** 2)
            total_diff += diff
        return total_diff / c

    @staticmethod
    def calculate_all_metrics(original_arr, stego_arr, payload_bytes, extracted_bytes=None):
        mse_val = StegoEvaluator.calculate_mse(original_arr, stego_arr)
        psnr_val = StegoEvaluator.calculate_psnr(original_arr, stego_arr)
        snr_val = StegoEvaluator.calculate_snr(original_arr, stego_arr)
        ssim_val = StegoEvaluator.calculate_ssim(original_arr, stego_arr)
        epi_val = StegoEvaluator.calculate_epi(original_arr, stego_arr)
        entropy_orig = StegoEvaluator.calculate_entropy(original_arr)
        entropy_stego = StegoEvaluator.calculate_entropy(stego_arr)
        capacity_bpp = StegoEvaluator.embedding_capacity(payload_bytes, original_arr.shape)
        fdm_val = StegoEvaluator.calculate_fdm(original_arr, stego_arr)

        norm_psnr = min(psnr_val / 60, 1.0)
        sei = (ssim_val * 0.4) + (norm_psnr * 0.3) + (epi_val * 0.3)
        sei_percent = round(sei * 100, 2)

        return {
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "snr": round(snr_val, 2),
            "mse": round(mse_val, 6),
            "capacity_bpp": round(capacity_bpp, 6),
            "fdm": round(fdm_val, 8),
            "epi": round(epi_val, 4),
            "entropy_original": round(entropy_orig, 4),
            "entropy_stego": round(entropy_stego, 4),
            "sei": sei_percent
        }
