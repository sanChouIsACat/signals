import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import plot
def encode(img_path, wm_path, res_path, alpha):
    img = cv2.imread(img_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    wm_height, wm_width = watermark.shape[0], watermark.shape[1]
    x, y = list(range(int(height / 2))), list(range(width))
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(height / 2)):
        for j in range(width): 

            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = watermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]
    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def decode(ori_path, img_path, res_path, alpha):
    ori = cv2.imread(ori_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (ori.shape[1], ori.shape[0]))
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)
    height, width = ori.shape[0], ori.shape[1]
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    random.seed(height + width)
    x = list(range(int(height / 2)))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)
    for i in range(int(height / 2)):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# image = cv2.imread('encoded.jpg')
# plot.plot_picture_feq(image, 'uncode',True)
# plt.show()
# exit()
decode('demo.jpg', 'snapshoot.jpg', 'snapshoot_decoded.jpg', 0.1)
exit()
# 读取第一张图片
img1 = cv2.imread('demo.jpg', cv2.IMREAD_GRAYSCALE)

# 读取第二张图片
img2 = cv2.imread('print.jpg', cv2.IMREAD_GRAYSCALE)

# 将第二张图片调整为与第一张图片相同的大小
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 对两张图片进行傅里叶变换
f1 = np.fft.fft2(img1)
f2 = np.fft.fft2(img2_resized)

# 将傅里叶变换结果移动到频域中心
f1_shifted = np.fft.fftshift(f1)
f2_shifted = np.fft.fftshift(f2)

# 计算振幅谱
magnitude1 = np.abs(f1_shifted)
magnitude2 = np.abs(f2_shifted)

# 将对应频率的振幅相加
combined_magnitude = magnitude1 + magnitude2

# 反傅里叶变换
combined_f_shifted = np.fft.ifftshift(combined_magnitude)
combined_img = np.fft.ifft2(combined_f_shifted)

# 取实部并归一化
combined_img = np.abs(combined_img)
combined_img = cv2.normalize(combined_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Image 1')
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Resized Image 2')
plt.imshow(img2_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Combined Magnitude')
plt.imshow(combined_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
