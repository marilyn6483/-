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

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)
            # return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def J(X_b, y, theta):

            """
            :param X_b:
            :param y:
            :param theta:
            :return: 损失函数
            """
            try:
                # print(np.sum((y - X_b.dot(theta)) ** 2) / len(X_b))
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def DJ(X_b, y, theta):

            """
            :param X_b: 处理之后的矩阵
            :param y: y值
            :param theta: θ
            :return res: 梯度向量
            """
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))
            return res * 2 / len(X_b)

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
                print(theta)
                if (abs(J(X_b, y, theta) - J(X_b, y, last_theta)) < eplison):
                    break
                n = n + 1
                # print(theta)
                print(n)
            return theta


        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def __repr__(self):
        return 'LinearRegression{}'


if __name__ == "__main__":
    # 创建一个线性相关得矩阵，元素个数100
    x = 2 * np.random.random(size=100)
    y = 3 * x + 4 + np.random.normal(size=100)
    x = x.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit_gd(x, y)
    # 给矩阵加一列常量为1的列
    print(lr._theta)