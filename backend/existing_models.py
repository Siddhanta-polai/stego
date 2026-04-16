import numpy as np
from PIL import Image
import hashlib

try:
    import pywt
    DWT_AVAILABLE = True
except ImportError:
    DWT_AVAILABLE = False

def _permute(password, n):
    seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], "big")
    rng = np.random.RandomState(seed)
    p = list(range(n))
    for i in range(n-1, 0, -1):
        j = rng.randint(0, i+1)
        p[i], p[j] = p[j], p[i]
    return p

class MSBModel:
    @staticmethod
    def embed(cover_arr, payload_bytes, password):
        stego = cover_arr.copy()
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        h, w, c = cover_arr.shape
        total_pixels = h * w * c
        perm = _permute(password, total_pixels)
        idx = 0
        for pos in perm:
            if idx >= len(bits):
                break
            i = (pos // (w * c)) % h
            j = (pos // c) % w
            ch = pos % c
            val = stego[i, j, ch]
            stego[i, j, ch] = (val & 0xFE) | bits[idx]
            idx += 1
        return stego

    @staticmethod
    def extract(stego_arr, length_bytes, password):
        h, w, c = stego_arr.shape
        total_pixels = h * w * c
        perm = _permute(password, total_pixels)
        bits = []
        for pos in perm[:length_bytes*8]:
            i = (pos // (w * c)) % h
            j = (pos // c) % w
            ch = pos % c
            bits.append(stego_arr[i, j, ch] & 1)
        data = bytearray()
        for k in range(0, len(bits), 8):
            if k+8 > len(bits):
                break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)

class DCTJStegModel:
    @staticmethod
    def embed(cover_arr, payload_bytes, password):
        from scipy.fftpack import dct, idct
        stego = cover_arr.copy()
        ycbcr = np.array(Image.fromarray(cover_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        h, w = y.shape
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        blocks_i = h // 8
        blocks_j = w // 8
        coeff_positions = []
        for bi in range(blocks_i):
            for bj in range(blocks_j):
                for u in range(1, 8):
                    for v in range(1, 8):
                        if 4 <= u+v <= 10:
                            coeff_positions.append((bi, bj, u, v))
        perm = _permute(password, len(coeff_positions))
        bit_idx = 0
        for pos in perm:
            if bit_idx >= len(bits):
                break
            bi, bj, u, v = coeff_positions[pos]
            block = y[bi*8:(bi+1)*8, bj*8:(bj+1)*8].copy()
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_int = np.round(dct_block).astype(np.int32)
            dct_int[u, v] = (dct_int[u, v] & 0xFE) | bits[bit_idx]
            bit_idx += 1
            dct_block = dct_int.astype(np.float32)
            block_new = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            y[bi*8:(bi+1)*8, bj*8:(bj+1)*8] = np.clip(block_new, 0, 255)
        ycbcr[:,:,0] = y
        return np.array(Image.fromarray(ycbcr, 'YCbCr').convert('RGB'))

    @staticmethod
    def extract(stego_arr, length_bytes, password):
        from scipy.fftpack import dct, idct
        ycbcr = np.array(Image.fromarray(stego_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        h, w = y.shape
        blocks_i = h // 8
        blocks_j = w // 8
        coeff_positions = []
        for bi in range(blocks_i):
            for bj in range(blocks_j):
                for u in range(1, 8):
                    for v in range(1, 8):
                        if 4 <= u+v <= 10:
                            coeff_positions.append((bi, bj, u, v))
        perm = _permute(password, len(coeff_positions))
        bits = []
        for pos in perm[:length_bytes*8]:
            bi, bj, u, v = coeff_positions[pos]
            block = y[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_int = np.round(dct_block).astype(np.int32)
            bits.append(dct_int[u, v] & 1)
        data = bytearray()
        for k in range(0, len(bits), 8):
            if k+8 > len(bits):
                break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)

class DWTHaarModel:
    @staticmethod
    def embed(cover_arr, payload_bytes, password):
        if not DWT_AVAILABLE:
            raise RuntimeError("PyWavelets not installed")
        stego = cover_arr.copy()
        ycbcr = np.array(Image.fromarray(cover_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        coeffs = pywt.dwt2(y, 'haar')
        LL, (LH, HL, HH) = coeffs
        bits = []
        for b in payload_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        total_coeff = LL.size
        perm = _permute(password, total_coeff)
        LL_flat = LL.flatten()
        for idx in perm[:len(bits)]:
            LL_flat[idx] = (LL_flat[idx] & 0xFE) | bits[idx]
        LL_new = LL_flat.reshape(LL.shape).astype(np.float32)
        y_new = pywt.idwt2((LL_new, (LH, HL, HH)), 'haar')
        y_new = np.clip(y_new, 0, 255).astype(np.uint8)
        ycbcr[:,:,0] = y_new
        return np.array(Image.fromarray(ycbcr, 'YCbCr').convert('RGB'))

    @staticmethod
    def extract(stego_arr, length_bytes, password):
        if not DWT_AVAILABLE:
            raise RuntimeError("PyWavelets not installed")
        ycbcr = np.array(Image.fromarray(stego_arr).convert('YCbCr'))
        y = ycbcr[:,:,0].astype(np.float32)
        coeffs = pywt.dwt2(y, 'haar')
        LL, _ = coeffs
        total_coeff = LL.size
        perm = _permute(password, total_coeff)
        LL_flat = LL.flatten()
        bits = []
        for idx in perm[:length_bytes*8]:
            bits.append(int(LL_flat[idx]) & 1)
        data = bytearray()
        for k in range(0, len(bits), 8):
            if k+8 > len(bits):
                break
            byte = 0
            for b in bits[k:k+8]:
                byte = (byte << 1) | b
            data.append(byte)
        return bytes(data)
