#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

def get_train_data():
    df = pd.read_csv('./train.csv')
    df = df[df['observation']=='PM2.5']
    df = df.iloc[:,3:]
    train_x = []
    train_y = []
    for i in range(15):
        tmp_x = df.iloc[:,i:i+9]
        tmp_x.columns = np.array(range(9))
        train_x.append(tmp_x)

        tmp_y = df.iloc[:,i+9]
        tmp_y.columns = np.array(range(1))
        train_y.append(tmp_y)
    train_x = pd.concat(train_x)
    train_y = pd.concat(train_y)
    train_x = np.array(train_x, float)
    train_y = np.array(train_y, float)

    return train_x, train_y

def get_test_data():
    df = pd.read_csv('./test.csv')
    df = df[df['AMB_TEMP']=='PM2.5'].iloc[:,2:]
    X = np.array(df, float)

    df_y = pd.read_csv('./answer.csv')
    y = df_y.value
    return X, y

def r2_score(y_true, y_predict):
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    return 1 - MSE / np.var(y_true)

class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=200):
        self.w = 0
        self.iter = n_iter
        self.eta = eta

    def net_input(self, X):
        return np.dot(X, self.w[1:].T) + self.w[0].T

    def adagrad(self, X, y):
        X = np.hstack([np.ones((len(X), 1)), X])
        self.w = np.zeros(X.shape[1])
        lr = 10
        sum_grad = np.zeros(X.shape[1])
        for i in range(self.iter):
            loss = np.dot(X, self.w) - y
            grad = 2*np.dot(X.T, loss)
            sum_grad += grad**2
            ada = np.sqrt(sum_grad)
            self.w -= lr*grad/ada
        print(self.w)
    def BGD(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.w = np.zeros(X_b.shape[1])
        m = X.shape[0]
        loss_his = []
        for i in range(self.iter):
            loss = np.dot(X_b, self.w) - y
            grad = np.dot(X_b.T, loss) / m
            self.w -= self.eta*grad
            loss_his.append(loss)

    def SGD(self,X,y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.w = np.zeros(1 + X.shape[1])
        m = X.shape[0]
        for i in range(self.iter):
            for i in range(m):
                rand_ind = np.random.randint(0, m)
                output = np.dot(X_b[rand_ind], self.w)
                loss = (y[rand_ind] - output)
                self.w -= self.eta * X_b[i].T.dot(loss)

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return np.dot(X_b, self.w)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)


train_X, train_y = get_train_data()
test_X, test_y = get_test_data()

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_X, train_y)
print(clf.coef_)
print('sklearn:', clf.score(test_X, test_y))

clf = LinearRegressionGD()
w1 = clf.adagrad(train_X, train_y)
y_pred = clf.predict(test_X)

print(clf.score(test_X, test_y))

clf = LinearRegressionGD()
w1 = clf.BGD(train_X, train_y)
y_pred = clf.predict(test_X)
print(clf.score(test_X, test_y))

clf = LinearRegressionGD()
w1 = clf.SGD(train_X, train_y)
y_pred = clf.predict(test_X)
print(clf.score(test_X, test_y))
