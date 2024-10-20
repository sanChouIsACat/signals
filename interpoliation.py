import numpy as np

def bilinear_interpolation(target_x: np.ndarray, target_y: np.ndarray, sampled_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 进行重采样
    standard_x = target_x.size
    standard_y = target_y.size
    resampled_signal = np.zeros((standard_x, standard_y))
    origin_x, origin_y = sampled_signal.shape

    x_ratio = float(origin_x / standard_x)
    y_ratio = float(origin_y / standard_y) 

    for i in range(standard_x):
        for j in range(standard_y):
            # 在原始网格中的坐标（浮点数）
            x = i * x_ratio
            y = j * y_ratio

            # 原始网格中相邻的4个整点
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, origin_x - 1)
            y2 = min(y1 + 1, origin_y - 1)

            # 获取4个顶点的值
            Q11 = sampled_signal[x1, y1]
            Q12 = sampled_signal[x1, y2]
            Q21 = sampled_signal[x2, y1]
            Q22 = sampled_signal[x2, y2]

            # 计算权重
            x2_x = x2 - x
            x_x1 = x - x1
            y2_y = y2 - y
            y_y1 = y - y1

            # 在水平方向上插值
            R1 = Q11 * x2_x + Q21 * x_x1
            R2 = Q12 * x2_x + Q22 * x_x1

            # 在垂直方向上插值
            P = R1 * y2_y + R2 * y_y1

            # 将插值结果存入新的数组
            resampled_signal[i, j] = P
    X, Y = np.meshgrid(target_x, target_y)
    return X, Y, np.atleast_2d(resampled_signal)