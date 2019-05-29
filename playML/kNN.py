import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class KNNClassifier:

    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        self._X_train = x_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict = [self._predict(x_predict) for x_predict in X_predict]

        return np.array(y_predict)

    def _predict(self, x_predict):
        distance = [sqrt(np.sum((x - x_predict) ** 2)) for x in self._X_train]
        nearest = np.argsort(distance)
        topK_y = self._y_train[nearest[:self.k]]
        votes = Counter(topK_y)
        y_predict = votes.most_common(1)[0][0]
        return y_predict

    def score(self, X_test, y_test):
        """
        根据测试数据集计算模型的准确度
        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return 'KNN(k={})'.format(self.k)

