import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        self.mean_ = np.array([np.mean(x[:, i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:, i]) for i in range(x.shape[1])])
        return self

    def transform(self, x):
        """
        均值归一化
        :return:
        """
        assert self.mean_ is not None and self.scale_ is not None, 'must fit befor transform'
        result = np.empty(shape=x.shape, dtype=float)
        target_num = x.shape[1]
        for i in range(target_num):
            result[:, i] = (x[:, i] - self.mean_[i]) /  self.scale_[i]

        return result