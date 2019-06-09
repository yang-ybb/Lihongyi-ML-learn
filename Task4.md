简介：

贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。本章首先介绍贝叶斯分类算法的基础——贝叶斯定理。最后，我们通过实例来讨论贝叶斯分类的中最简单的一种: 朴素贝叶斯分类。

模型假设：

我们假设特征之间 相互条件独立 。换句话来说就是特征向量中一个特征的取值并不影响其他特征的取值。所谓 独立(independence) 指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系，比如说，“我们”中的“我”和“们”出现的概率与这两个字相邻没有任何关系。这个假设正是朴素贝叶斯分类器中 朴素(naive) 一词的含义。朴素贝叶斯分类器中的另一个假设是，每个特征同等重要。

模型思想：

通过数据调整先验分布，得到每个样本属于每个类的条件概率，预测值为最大的条件概率所代表的的类。

模型公式： $$ P(c | \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} | c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^{d} P\left(x_{i} | c\right) $$

$$ h_{n b}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^{d} P\left(x_{i} | c\right) $$

$$ P(c)=\frac{\left|D_{\mathrm{c}}\right|}{|D|} $$

$$ P\left(x_{i} | c\right)=\frac{\left|D_{c, x_{i}}\right|}{\left|D_{c}\right|} $$

假定$p\left(x_{i} | c\right) \sim \mathcal{N}\left(\mu_{c, i}, \sigma_{c, i}^{2}\right)$ $$ p\left(x_{i} | c\right)=\frac{1}{\sqrt{2 \pi} \sigma_{c, i}} \exp \left(-\frac{\left(x_{i}-\mu_{c, i}\right)^{2}}{2 \sigma_{c, i}^{2}}\right) $$

优化方法：

按步骤计算。

优缺点

优点: 在数据较少的情况下仍然有效，可以处理多类别问题。 缺点: 对于输入数据的准备方式较为敏感。 适用数据类型: 标称型数据。
