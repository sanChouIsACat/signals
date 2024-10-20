import numpy as np
from interpoliation import *
from plot import *
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import filter
from global_var import *
freq_x, freq_y = 10, 10  # 频率
sample_feq = 80  # 采样率

def fft2d_draw(signal):
    fft_res = np.fft.fft2(signal)
    fft_res = np.fft.fftshift(fft_res)
    fft_res = np.abs(fft_res)
    return fft_res

def generate_2d_signal(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成一个二维正弦波信号"""

    Z = np.sin(2 * np.pi * freq_x * x) * np.sin(2 * np.pi * freq_y * y) # 生成二维信号
    return x , y, Z

def sample_signal(signal, sample_feq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对信号进行采样"""
    x = np.linspace(0, x_size,x_size * sample_feq, endpoint=False)
    y = np.linspace(0, y_size,y_size * sample_feq, endpoint=False)
    X, Y = np.meshgrid(x, y)  # 创建网格
    
    # 进行采样
    sampled_signal = signal(X, Y)[2]
    return X,Y,sampled_signal

# 生成二维信号
_, __, signal = generate_2d_signal(plot_X, plot_Y)

# 对信号进行采样
sampled_x, sampled_y, sampled_signal = sample_signal(generate_2d_signal, sample_feq)

# 各种过滤器
# 低通过滤器
low_pass_filter_ftt, low_pass_filter_signal = filter.low_pass_filter(sampled_signal, 40)
# 高斯过滤器
kernel = np.random.randn(20, 20)
convolved_signal = convolve2d(sampled_signal, kernel, mode='same', boundary='wrap')

# 重采样以显示
_, __, resampled_convolved_signal = bilinear_interpolation(plot_x, plot_y, convolved_signal)
_, __, resampled_signal = bilinear_interpolation(plot_x, plot_y, sampled_signal)
_, __, resampled_lowpass_signal = bilinear_interpolation(plot_x, plot_y, low_pass_filter_signal)
standard_resampled_signal = griddata((sampled_x.ravel(), sampled_y.ravel()), sampled_signal.ravel(), (plot_X, plot_Y), method='linear')
# 在频域中重建信号

# 绘图
# 原始信号和频域
plot_signal_and_frd(plot_X, plot_Y, signal, fft2d_draw(signal), 'X axis', 'Y axis', 'Original')

# # 采样信号
plot_signal_and_frd(sampled_x, sampled_y, sampled_signal, fft2d_draw(sampled_signal), 'X axis', 'Y axis', 'sample')

# 标准重采样信号
plot_signal_and_frd(plot_X, plot_Y, standard_resampled_signal, fft2d_draw(standard_resampled_signal), 'X axis', 'Y axis', 'Standard Resampled(NF)')
# 没有应用滤波器的重建信号
plot_signal_and_frd(plot_X, plot_Y, resampled_signal, fft2d_draw(resampled_signal), 'X axis', 'Y axis', 'Manual Resampled(NF)')

# 应用高斯滤波器的重建信号
plot_signal_and_frd(plot_X, plot_Y, resampled_convolved_signal, fft2d_draw(resampled_convolved_signal), 'X axis', 'Y axis', 'Manual Resampled(GS)')
# 应用低通过滤波器的重建信号
plot_signal_and_frd(plot_X, plot_Y, resampled_lowpass_signal, fft2d_draw(resampled_lowpass_signal), 'X axis', 'Y axis', 'Manual Resampled(LP)')
# 应用低通过滤波器的频域
plot_signal_and_frd(sampled_x,sampled_y,low_pass_filter_signal, np.abs(low_pass_filter_ftt), 'X axis', 'Y axis', 'Low Pass Filter')
# 滤波器的频域
plot_filter(fft2d_draw(kernel), 'X axis', 'Y axis', 'Kernel')

