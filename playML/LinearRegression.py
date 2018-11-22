import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        # 使用正规化方程进行训练
        assert x_train.shape[0] == y_train.shape[0]
        x_b = np.hstack((np.ones((len(x_train), 1)), x_train))
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def predict(self, x_predict):
        assert self.coef_ is not None and self.interception_ is not None
        assert x_predict.shape[1] == len(self.coef_)
        x_b = np.hstack((np.ones((len(x_predict), 1)), x_predict))
        return x_b.dot(self._theta)

    # def score(self):

    def __repr__(self):
        return 'LinearRegression{}'