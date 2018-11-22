import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        self.a_ = np.sum((x_mean - x) * (y_mean - y )) / np.sum((x - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        # 返回np数组
        return np.array(y_predict)

    def _predict(self, x):
        return self.a_ * x + self.b_


class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        # print((x_mean - x).dot(y_mean - y))
        # print((x - x_mean).dot(x - x_mean))
        self.a_ = (x_mean - x).dot(y_mean - y) / (x - x_mean).dot(x - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        # 返回np数组
        return np.array(y_predict)

    def _predict(self, x):
        return self.a_ * x + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return self.r2_score(y_test, y_predict)

    def r2_score(self, y_test, y_predict):
        return 1 - np.sum((y_predict - y_test) ** 2) / np.sum((np.mean(y_test) - y_test) ** 2)


if __name__ == "__main__":
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([1., 3., 2., 3., 5.])
    slr = SimpleLinearRegression1()
    slr.fit(x, y)
    print(slr.a_)