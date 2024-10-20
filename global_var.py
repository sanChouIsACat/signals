import numpy as np

origin_sample_feq = 1000
x_size, y_size = 1, 1  # 信号的大小
plot_x = np.linspace(0, x_size, x_size * origin_sample_feq, endpoint= False)
plot_y = np.linspace(0, y_size, y_size * origin_sample_feq, endpoint= False)
plot_X, plot_Y = np.meshgrid(plot_x, plot_y)  # 创建网格