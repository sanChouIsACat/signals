import numpy as np
from interpoliation import *
from plot import *
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from filter import *
from global_var import *
from PIL import Image
from plot import *
import cv2


image = cv2.imread('demo.src', cv2.IMREAD_COLOR)

hight, width = image.shape[:2]
red_channel = image[:, :, 0]    # 红色通道
green_channel = image[:, :, 1]  # 绿色通道
blue_channel = image[:, :, 2]   # 蓝色通道
plot_picture(image, 'ori_demo')
plot_picture_feq(image, 'ori_demo_feq')

def filter_picture(picuture, filter):
    res = []
    channles = cv2.split(picuture)
    for channel in channles:
        channel_fft = fft2(channel)
        channel_fft = apply_filter(channel_fft, filter)
        channel_fft = ifft2(channel_fft).real
        channel_fft = np.abs(channel_fft)
        channel_fft = cv2.normalize(channel_fft, None, 0, 255, cv2.NORM_MINMAX)
        channel_fft = np.uint8(channel_fft)
        res.append(channel_fft)
    res = tuple(res)
    return cv2.merge(res)

# filter
gray_channel = 0.2989 * red_channel + 0.5870 * green_channel + 0.1140 * blue_channel
gray_fft = np.fft.fft2(gray_channel)
draw_feq = np.fft.fftshift(gray_fft)

# low pass filter
low_pass_filter = get_low_pass_filter(gray_fft, 100)
low_pass_filter_pig = filter_picture(image, low_pass_filter)
cv2.imwrite('low_pass_pig.jpg', low_pass_filter_pig)
plot_filter_res(gray_fft, low_pass_filter, apply_filter(gray_fft, low_pass_filter), 'filter(LFT)')
plot_picture_feq(low_pass_filter_pig, 'picture(LFQ)')

# high pass filter
high_pass_filter = get_high_pass_filter(gray_fft, 40)
high_pass_filter_pig = filter_picture(image, high_pass_filter)
cv2.imwrite('high_pass_pig.jpg', high_pass_filter_pig)
plot_filter_res(gray_fft, high_pass_filter, apply_filter(gray_fft, high_pass_filter), 'filter(HFT)')
plot_picture_feq(high_pass_filter_pig, 'picture(HFQ)')

# sharp
sharp_pass_filter = get_gaussian_sharp_filter(gray_fft, 200)
sharp_pass_filter_pig = filter_picture(image, sharp_pass_filter)
cv2.imwrite('sharp_pass_pig.jpg', sharp_pass_filter_pig)
plot_filter_res(gray_fft, sharp_pass_filter, apply_filter(gray_fft, sharp_pass_filter), 'filter(Sharp)')
plot_picture_feq(sharp_pass_filter_pig, 'picture(Sharp)')

# sharp2
result = cv2.addWeighted(image, 1, high_pass_filter_pig, 10, 0)
cv2.imwrite('sharp_by_add.jpg', sharp_pass_filter_pig)
plot_picture_feq(sharp_pass_filter_pig, 'picture(Sharp_by_add)')


