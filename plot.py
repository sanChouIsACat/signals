import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cv2

def fft2d_draw(signal):
    fft_res = np.fft.fft2(signal)
    fft_res = np.fft.fftshift(fft_res)
    fft_res = np.abs(fft_res)
    return np.log(fft_res + 1)

red_cmap = LinearSegmentedColormap.from_list('custom_red', ['white', 'red'])
green_cmap = LinearSegmentedColormap.from_list('custom_red', ['white', 'green'])
blue_cmap = LinearSegmentedColormap.from_list('custom_red', ['white', 'blue'])
def plot_picture(picture, title):
    gray_array = 0.2989 * picture[:, :, 0] + 0.5870 * picture[:, :, 1] + 0.1140 * picture[:, :, 2]
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(picture[:,:,0], cmap=red_cmap)
    plt.subplot(2, 2, 2)
    plt.imshow(picture[:,:,1], cmap=green_cmap)
    plt.subplot(2, 2, 3)
    plt.imshow(picture[:,:,2], cmap=blue_cmap)
    plt.subplot(2, 2, 4)
    plt.imshow(gray_array, cmap='gray')
    plt.tight_layout()
    plt.savefig('{}.jpg'.format(title), format='jpg', dpi=300)  # dpi 可根据需要调整

def plot_picture_feq(picture: np.ndarray, title, from_cv2=False):
    if from_cv2:
        r,g,b = cv2.split(picture)
    else:
        r,g,b = picture[:,:,0], picture[:,:,1], picture[:,:,2]
    gray_array = 0.2989 * r + 0.5870 * g + 0.1140 * b
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(picture)
    plt.subplot(1, 2, 2)
    plt.imshow(fft2d_draw(gray_array), cmap='gray')
    plt.tight_layout()
    plt.savefig('{}.jpg'.format(title), format='jpg', dpi=300)  # dpi 可根据需要调整

def plot_2d_signal(x, y, signal, x_label, y_label, title):
    plt.scatter(x.flatten(),y.flatten(),c = signal)
    plt.title(title + ' (Time Domain)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()

def plot_2d_frq(feq, x_label, y_label, title):
    plt.imshow(feq + 1, cmap='gray') 
    plt.title(title + ' (Frequency Domain)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()

def plot_2d_frq_origin(feq, x_label, y_label, title):
    draw_feq = np.fft.fftshift(feq)
    plt.title(title)
    plt.imshow(np.log(np.abs(draw_feq) + 1), cmap='gray') 

def plot_signal_and_frd(x, y, signal, feq, x_label, y_label, title):
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_2d_signal(x, y, signal, x_label, y_label, title)
    plt.subplot(1, 2, 2)
    plot_2d_frq(feq, x_label, y_label, title)
    plt.savefig('{}.jpg'.format(title), format='jpg', dpi=300)  # dpi 可根据需要调整

def plot_filter(feq, x_label, y_label, title):
    plt.figure()
    plt.subplot(1, 1, 1)
    plot_2d_frq(feq, x_label, y_label, title)
    plt.savefig('{}.jpg'.format(title), format='jpg', dpi=300)  # dpi 可根据需要调整

def plot_2d_frq2(feq1, feq2, x_label, y_label, all_title, title1, title2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_2d_frq(feq1, x_label, y_label, title1)
    plt.subplot(1, 2, 2)
    plot_2d_frq(feq2, x_label, y_label, title2)
    plt.savefig('{}.jpg'.format(all_title), format='jpg', dpi=300)  # dpi 可根据需要调整

def plot_filter_res(feq1, filter, feq2, title):
    plt.figure()
    plt.subplot(2, 2, 1)
    plot_2d_frq_origin(feq1, "", "", title+"(before)")
    plt.subplot(2, 2, 2)
    plot_2d_frq_origin(filter, "", "", "filter")
    plt.subplot(2, 2, 3)
    plot_2d_frq_origin(feq2, "", "", title+"(after)")
    plt.savefig('{}.jpg'.format(title), format='jpg', dpi=300)  # dpi 可根据需要调整