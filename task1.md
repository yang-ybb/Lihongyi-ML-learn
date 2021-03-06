1. 中心极限定理：
1）任何一个样本的平均值将会约等于其所在总体的平均值。
2）不管总体是什么分布，任意一个总体的样本平均值都会围绕在总体的平均值周围，并且呈正态分布。
作用：用来通过采样来估计总体的情况；还可以根据总体的平均值和标准差来判断某一个样本是否属于该总体。
2. Loss Function
视频中的例子是线性回归，方程是：$y=b+\sum w_ix_i$，误差是用真实值与预测值之差的平方：$L(w,b )=\sum_{n=1}^N(y^n-(b+\sum w_ix_i))^2$
3. 损失函数与凸函数之间的关系
最小二乘法是线性回归的一种，OLS将问题转化成了一个凸优化问题。在线性回归中，它假设样本和噪声都服从高斯分布，最后通过极大似然估计（MLE）可以推导出最小二乘式子。最小二乘的基本原则是：最优拟合直线应该是使各点到回归直线的距离和最小的直线，即平方和最小。
4. 全局最优和局部最优
因为损失函数是凸函数，那么局部最优解就是全局最优解。
基于梯度得搜索是使用最为广泛得参数寻优方法。在此类方法中，我们从某些初始解出发，迭代寻找最优参数值。每次迭代中，我们先计算误差函数在当前点的梯度，然后根据梯度确定搜索方向。例如，由于负梯度方向是函数值下降最快的方法，因此梯度下降法就是沿着负梯度方向搜索最优解。若误差函数在当前点的梯度为零，则已达到局部极小，更新量将为零，这意味着参数的迭代更新将在此停止。显然，如果误差函数仅有一个局部极小，那么此时找到的局部极小就是全局最小。然后，如果误差函数具有多个局部极小，则不能保证找到的解是全局最小。对后一种情形，我们称参数寻优陷入了局部极小，这显然不是我们所希望的。
5. 推导梯度下降公式
![image.png](https://upload-images.jianshu.io/upload_images/3248297-91344dcf61ae83cf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
6. 梯度下降的代码
```python
x_train = [1,5,7,8]
y_train = [1,9,15,17]
alpha = 0.00001
diff = [0, 0]
cnt = 0
m = len(x_train)
theta0 = 0
theta1 = 0
error0=0
error1=0
epsilon=0.000001

def h(x):
    return theta1*x+theta0
while True:
    cnt=cnt+1
    diff = [0,0]
    for i in range(m):
        diff[0]+=h(x_train[i])-y_train[i]
        diff[1]+=(h(x_train[i])-y_train[i])*x_train[i]
    theta0=theta0-alpha/m*diff[0]
    theta1=theta1-alpha/m*diff[1]

    error1=0
    for i in range(len(x_train)):
        error1 += (y_train[i] - (theta0 + theta1 * x_train[i])) ** 2 / 2

    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
```
7. L0、L1正则
模型越复杂，对于验证集的效果越好，但是对于测试集效果就变差（过拟合），所以需要在损失函数后面加上一个正则化项来约束变量不能太多。$L(w,b )=\sum_{n=1}^N(y^n-(b+\sum w_ix_i))^2+λ\sum (w_i)^2$，这个是L1正则。
L0范数是指向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。
 L1范数是指向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。
L2范数: ||W||2
L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解。
8. 学习为什么只对w/Θ做限制，不对b做限制
因为b影响的只是函数的上下平移，对函数本身的复杂性没有影响。过拟合是因为函数本身的复杂性造成的。
