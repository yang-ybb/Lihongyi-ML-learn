class LogisticRegression():
    '''
    :param lr: 学习率
    :param num_iters: 更新轮数
    :param seed: 随机数种子
    '''
    def __init__(self, lr=0.1, num_iters=100, seed=None):
        self.seed = seed
        self.lr = lr
        self.num_iters = num_iters


    def fit(self, x, y):
        np.random.seed(self.seed)
        # 参数初始化w b
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        # 数据集
        self.x = x
        self.y = y
        # 迭代更新
        for i in range(self.num_iters):
            self._update_step()

    
    # sigmod处理
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    # 函数模型 w*x + b，经过SIGMOD处理
    def _f(self, x, w, b):
        z = x.dot(w) + b
        return self._sigmoid(z)


    # 初次预测算出概率        
    def predict_proba(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred


    # 再预测，根据概率分类
    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred


    # 为分类进行评分
    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        # 计算准确率            
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc


    # 损失函数
    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))


    # 梯度下降
    def gradient_descent(self):
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b
        
        
