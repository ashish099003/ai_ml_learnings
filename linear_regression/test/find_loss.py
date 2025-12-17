
import numpy as np


def cost(x, y, w):
    x = np.asarray(x)
    w = np.asarray(w)
    y = np.asarray(y)

    y_pred = x @ w

    n = len(y)
    loss = (1 / (2 * n)) * np.sum((y - y_pred) ** 2)
    return round(loss, 2)

def calculate_r_square(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    print(y_true)
    print(y_pred)
    y_mean = y_true.mean()
    print(y_mean)
    ss_total =np.sum((y_true-np.mean(y_true))**2)
    print(ss_total)
    ss_residual = np.sum((y_true-y_pred)**2)
    print(ss_residual)
    r2 = 1- (ss_residual/ss_total)
    print(r2)


if __name__=='__main__':
    x = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [1.0, 8.0],
         [1.0, 9.0]]
    y = [32.69, 30.56, 32.04, 32.27, 35.98, 34.39, 40.25, 39.24, 37.14, 39.38]
    w = [30.0, 1.0]
    # print(cost(x,y,w))
    y_true = [9.91, 12.79, 12.36, 5.31, 4.11]
    y_pred = [0.88, 2.6, 0.36, -0.45, 1.64]
    calculate_r_square(y_true,y_pred)