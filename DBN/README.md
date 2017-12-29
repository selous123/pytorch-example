

## RBM(Restricted Boltzmann Machine)

### RBM的基本结构
和一层神经网络(NN)的结构是一样的
![](/home/lrh/program/git/pytorch-example/DBN/rbm.png) 

其中输入层叫做v(可视层),输出层h(隐藏层),两者通过权重矩阵W相连.

** 1.能量函数 **
$$E(v,h) = -b^Tv-c^Th-v^TWh$$
** 2.指定联合分布 **
$$P(v,h) = \frac{1}{Z}exp(-E(v,h))$$
其中配分函数\\(Z = \sum_v\sum_hexp\\{-E(v,h)\\}\\);也就是枚举所有的状态,然后求和,复杂度为\\(O(2^{n_h+n_v})\\),易知该问题是难解的.
** 3.条件分布 **
$$p(h_j=1|V) = \sigma(c_j+v^TW_{:,j})$$
推导过程见笔记

** 结论:虽然联合分布和边缘分布是难解的,但是条件分布是容易解出来的.**


### 对数似然函数
假设训练样本集合$$S = \{v^1,v^2,\dots,v^{n_s}\}$$
其中$$v^i = (v_1^i,v_2^i,\dots,v_{n_v}^i),i=1,2,\dots,n_s$$

在已知\\(v\\)的分布的情况下,也就是\\(P(v)\\)

训练RBM的目标是:
$$\max L_{\theta,S} = \prod_{i=1}^{n_s}P(v^i)$$
转化成ln对数表达,
$$\max lnL_{\theta,S} = \sum_{i=1}^{n_s}lnP(v^i)$$

### 梯度优化
迭代
$$\theta^{t+1} = \theta^{t}+\eta\frac{dlnL_{\theta,S}}{d\theta}$$

梯度:
**[考虑单个样本的梯度计算]**
$$lnL_s = lnP(v) = ln(\frac{\sum_he^{-E(v,h)}}{Z}) = ln\sum_h e^{-E(v,h)}-ln\sum_{v,h}e^{-E(v,h)}$$
$$\frac{dlnp(v)}{dw_{ij}} = p(h_i=1|v)v_j-\sum_vp(v)p(h_i=1|v)v_j 
\\ = p(h_i=1|v)v_j-E_{v\sim p(v)}[p(h_i=1|v)*v_j]$$
$$\frac{dlnp(v)}{db_i} = v_i - \sum_vp(v)v_i \\
= v_i - E_{v\sim p(v)}[v_i]$$
$$\frac{dlnp(v)}{dc_i} = p(h_i=1|v) - \sum_vp(v)p(h_i=1|v)\\
=p(h_i=1|v) - E_{v \sim p(v)}[p(h_i = 1|v)]$$

**因为p(v)是难解的,所以该问题的复杂度也非常高**


### 对比散度



