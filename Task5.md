1.
LR逻辑回归是一种监督学习分类算法，其实现了给定数据集到0,1的一种映射。
给定数据集其中（xi,yi）表示第i个样本，其中。即每个数据有n个特征，类别，要求训练数据，将数据分成两类0或1。
假定xi的n个特征为线性关系，即：z=θx+b=θ1x1+θ2x2+...+θnxn+b
可以使用阶跃函数，但是阶跃函数性质不好，不可导求解过于复杂，这里选用Sigmoid函数:
y(z)=1/(1+e^(-z))
当输入一个Z时，y输出一个0--1之间的数，假定y>0.5则最终结果判为1  y<0.5最终结果为0。当y=0.8时，最终结果为1,y=0.8也表征了此时输出为1的概率

2.
![image.png](https://upload-images.jianshu.io/upload_images/3248297-f7889a465c90f14b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3.
Softmax回归模型是logistic回归模型在多分类问题上的推广，当分类数为2的时候会退化为Logistic分类。.在多分类问题中，类标签 y 可以取两个以上的值。 Softmax回归模型对于诸如MNIST手写数字分类等问题是很有用的，该问题的目的是辨识10个不同的单个数字。Softmax回归是有监督的。

4.
![image.png](https://upload-images.jianshu.io/upload_images/3248297-974da0623faa3c71.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

5.
![image.png](https://upload-images.jianshu.io/upload_images/3248297-7df6d2827101cfd2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
