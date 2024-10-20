import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的 1D 信号 (正弦波 + 噪声)
N = 500  # 样本数
t = np.linspace(0, 1, N, endpoint=False)  # 时间轴
freq = 5  # 正弦波的频率
signal = np.sin(2 * np.pi * freq * t)  # 带噪声的信号

# 计算 1D 傅里叶变换
signal_fft = np.fft.fft(signal)

# 频率轴
freqs = np.fft.fftfreq(N, d=t[1] - t[0])

# 将频谱移动，使得零频率成分在中心
signal_fft_shifted = np.fft.fftshift(signal_fft)
freqs_shifted = np.fft.fftshift(freqs)

# 计算频谱的幅值
spectrum = np.abs(signal_fft_shifted)

# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 绘制频谱
plt.subplot(2, 1, 2)
plt.plot(freqs_shifted, spectrum)
plt.title('Fourier Spectrum (Frequency Domain)')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
