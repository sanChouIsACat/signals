from scipy.fft import fft2, ifft2
import numpy as np

def in_circle(i: int, j:int, row:int, col:int, r:int) -> bool:
    left_len = i - 0
    top_len = j - 0
    right_len = row - i
    bottom_len = col - j
    return np.sqrt(left_len ** 2 + top_len ** 2) < r or \
        np.sqrt(right_len ** 2 + top_len ** 2) < r or \
            np.sqrt(left_len ** 2 + bottom_len ** 2) < r or \
                np.sqrt(right_len ** 2 + bottom_len ** 2) < r

def get_low_pass_filter(signal_fft: np.ndarray , threshold: float) -> np.ndarray:
    """低通滤波"""
    res = np.zeros_like(signal_fft)
    rows, cols = signal_fft.shape
    for i in range(rows):
        for j in range(cols):
            if in_circle(i, j, rows, cols,threshold):
                res[i, j] = 1 + 0j
    return res

def get_high_pass_filter(signal_fft: np.ndarray, threshold: float) -> np.ndarray:
    """高通滤波"""
    res = np.zeros_like(signal_fft)
    rows, cols = signal_fft.shape
    for i in range(rows):
        for j in range(cols):
            if not in_circle(i, j, rows, cols,threshold):
                res[i, j] = 1 + 0j
    return res

def apply_filter(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    res = signal.copy()
    """卷积"""
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            res[i, j] = kernel[i, j] * res[i, j]
    return res

def get_gaussian_sharp_filter(signal: np.ndarray, threshold: float):
    """
    创建高斯高通滤波器。
    
    :param shape: 输入图像的形状 (rows, cols)
    :param cutoff_frequency: 截止频率
    :return: 高斯高通滤波器
    """
    rows, cols = signal.shape
    crow, ccol = rows // 2, cols // 2

    # 创建高斯高通滤波器
    x = np.linspace(-ccol, ccol - 1, cols)
    y = np.linspace(-crow, crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 - np.exp(-(D**2) / (2 * (threshold**2)))  # 高斯高通响应

    return H