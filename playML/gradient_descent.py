import numpy as np


# 损失函数梯度
def DJ(X_b, y, theta):
    """
    :param X_b: 处理之后的矩阵
    :param y: y值
    :param theta: θ
    :return res: 梯度向量
    """
    res = np.empty(len(theta))
    res[0] = np.sum((X_b.dot(theta) - y))
    for i in range(1, len(theta)):
        res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))
    return res * 2 / len(X_b)


def J(X_b, y, theta):
    """
    :param X_b:
    :param y:
    :param theta:
    :return: 损失函数
    """
    return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)


def gradient_descent(X_b, y, init_theta, eta, n_iters=1e4, eplison=1e-8):
    """
    :param init_theta: theta初始值
    :param eta: 系数eta
    :param eplison: 迭代的临界值
    :param n_iters: 拟合的次数上限值
    :return theta: 拟合求得得theta向量
    """
    theta = init_theta
#     设定最大的学习次数
    n = 0
    while n < n_iters:
        gradient = DJ(X_b, y, theta)
        last_theta = theta
        theta = theta - eta * gradient
        # print("in")
        if abs(J(x_b, y, theta) - J(X_b, y, last_theta)) < eplison:
            break
        n = n + 1
        # print(theta)
        print(n)
    return theta


if __name__ == "__main__":
    # 创建一个线性相关得矩阵，元素个数100
    x = 2 * np.random.random(size=(1000, 1))
    y = 3 * x + 4 + np.random.normal(size=(1000, 1))
    # 给矩阵加一列常量为1的列
    x_b = np.hstack([np.ones((len(x), 1)), x])
    init_theta = np.zeros(x_b.shape[1])

    eta = 0.001
    # print(x_b)
    # print(J(x_b, y, init_theta))
    print(gradient_descent(x_b, y, init_theta, eta))