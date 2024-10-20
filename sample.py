import numpy as np
def sample_signal(signal, sample_feq, signals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对信号进行采样"""
    rows, cols = signal.shape
    x = np.linspace(0, x_size,x_size * sample_feq, endpoint=False)
    y = np.linspace(0, y_size,y_size * sample_feq, endpoint=False)
    X, Y = np.meshgrid(x, y)  # 创建网格
    
    # 进行采样
    sampled_signal = generate_2d_signal(X, Y)[2]
    return X,Y,generate_2d_signal(X, Y)