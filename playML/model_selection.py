import numpy as np


def train_test_split(x, y, ratio=0.2, seed=None):

    # 数据x和数据y的行数必须相等
    assert x.shape[0] == y.shape[0], "The size of x must be equal to y."
    assert 0.0 <= ratio <= 1.0, "Test ratio must be valid."

    if seed:
        np.random.random(seed)
    # len(x)是x的行数，生成随机索引向量
    shuffled_indexes = np.random.permutation(len(x))
    # 测试数据集的大小
    test_size = int(len(x) * ratio)
    train_indexes = shuffled_indexes[test_size:]
    test_indexes = shuffled_indexes[:test_size]
    x_train = x[train_indexes]
    y_train = y[train_indexes]
    x_test = x[test_indexes]
    y_test = y[test_indexes]

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x = np.random.random(size=1000)
    y = np.random.random(size=1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y)