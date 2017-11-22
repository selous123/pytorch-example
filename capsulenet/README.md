代码运行环境：
python2.7,pytoch0.4.0a0+6eca9e0,visdom0.1.6.0
数据集使用mnist数据集，解压之后的数据
# pytorch实现capsule

参考论文：https://arxiv.org/pdf/1710.09829.pdf

参考博客：http://blog.csdn.net/uwr44uouqcnsuqb60zk2/article/details/78463687

参考代码：https://github.com/timomernick/pytorch-capsule

## 写下开头
capsule是hinton老爷子最近提出的神经网络架构，其目的在于将传统的用点表示特征的方式改为使用向量表示．所以capsule网络的输出v的shape为[batch_size,10,16,1]，其中向量v_j([16,1])的范数就为对应类别的可能性P

## 基本概念
### loss function
每一个类的损失：
$$L_c = T_c\max(0,m^+-||v||)^2+\lambda(1-T_c)\max(0,||v||-m^-)^2$$
总损失为：
$$loss = \frac{1}{c}\sum_{i=0}^cL_i$$
```
def loss(labels,v):
    """
    input:
        labels:[batch_size,10]
        v:[batch_size,10,16,1]
    """
    #shape->[batch_size,10,1,1]
    #print v
    v_norm = torch.sqrt(torch.sum(v**2,dim=2,keepdim=True)).squeeze()
    zero = torch.zeros([1]).double()
    lamda = torch.Tensor([0.5]).double()
    if conf.cuda:
	zero = zero.cuda()
	lamda = lamda.cuda()
    zero = Variable(zero)
    lamda = Variable(lamda)
    m_plus = 0.9
    m_minus = 0.1
    #shape->[batch_size,10]
    L = torch.max(zero,m_plus-v_norm)**2
    R = torch.max(zero,v_norm-m_minus)**2
    #equation 4 in paper
    loss = torch.sum(labels*L+lamda*(1-labels)*R,dim=1)
    #shape->[batch_size,]
    loss = loss.mean()
    #shape->[1,]
    return loss
```

### activation function(squashing)

$$v_j = \frac{||s_j||}{1+||s_j||}\frac{s_j}{||s_j||}$$
易知$$p(pre\_y=j) = ||v_j||<1$$

```
def squash(x,dim):
    #we should do spuash in each capsule.
    #we use dim to select
    sum_sq = torch.sum(x**2,dim=dim,keepdim=True)
    sum_sqrt = torch.sqrt(sum_sq)
    return (sum_sq/(1.0+sum_sq))* x/sum_sqrt
```
## 网络架构
下面的网络架构主要通过输入的维度变化进行演示(一直没有找到好的画图的工具)
### conv
输入是input([batch_size,1,28,28])
选择kernel([1,256,9,9]),stride=1,
则第一层conv输出为conv_output([batch_size,256,20,20])

### capsconv
同样是卷积操作,input([batch_size,256,20,20])
选择kernel([256,32,9,9]),stride=2，卷积运算结果为conv_result([batch_size,32,6,6])
**完全独立的重复上面的操作８次，结果储存为list_conv_result[conv_1_result,...,conv_8_result]**
然后把这８个结果stack在一起，结果为capsconv_stack([batch_size,8,32,6,6]),然后将后三维合并在一起时候才能capsconv_output([batch_size,8,1152]),交换一下1，2维capsconv_output([batch_size,1152,8])，然后调用一次激活函数

```
#coding:utf-8
#author:selous
import torch
import torch.nn as nn
import utils

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)
    def forward(self, x):
        return self.conv0(x)
class CapsConv(nn.Module):
    def __init__(self,in_channels=256,out_dim=8):
        super(CapsConv,self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_channels)
            self.add_module("unit_" + str(unit_idx), unit)
            return unit
        #定义８次卷积操作
        self.conv = [create_conv_unit(i) for i in range(self.out_dim)]
    def forward(self,x):
        #input x with shape ->[batch_size,in_features,height,width]
        #output with shape->[batch_size,32,6,6]
        x = [self.conv[i](x) for i in range(self.out_dim)]
        #output with shape->[batch_size,8,32,6,6]
        x = torch.stack(x,dim=1)
        #return shape->[batch_size,1152,8]
        x = x.view(x.size(0),self.out_dim,-1).transpose(1,2)
        #return shape->[batch_size,1152,8]
        x = utils.squash(x,dim=2)
        return x

```
### capsnet
输入为input[batch_size,1152,8],首先stack成[batch_size,1152,10,1,8],然后构建一个全连接参数W[1,1152,10,8,16],堆叠成[batch_size,1152,10,8,16],两个点乘结果为mut_result[batch_size,1152,10,1,16]
下面就是要使用动态路由的算法，优化一个c[batch_size,1152,10,1,1]，先不考虑动态优化算法，假设我们已经优化好了c，用c[batch_size,1152,10,1,1]与mut_result[batch_size,1152,10,1,16]进行数乘之后关于第1维求和capsnet_result[batch_size,1,10,1,16]
然后调用一下激活函数输出结果capsnet_result[batch_size,10,16,1]

### 动态路由算法
首先初始化一个b([1,1152,10,1])，先关于第二维求softmax,然后结果stack成b_stack[batch_size,1152,10,1]，增加一维变成c[batch_size,1152,10,1,1],然后通过上面说的过程求capsnet_result[batch_size,1,10,1,16],利用这个结果更新b，先将capsnet_result变成[batch_size,1152,10,1,16],然后转成[batch_size,1152,10,16,1],然后与mut_result[batch_size,1152,10,1,16]进行点乘得到结果[batch_size,1152,10,1,1],然后关于第０维求平均结果squeeze变成[1,1152,10,1],然后直接更新b
循环三次得出最后结果


```
#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import config

conf = config.DefaultConf()
class CapsNet(nn.Module):
    """
    input :a group of capsule -> shape:[batch_size*1152(feature_num)*8(in_dim)]
    output:a group of new capsule -> shape[batch_size*10(feature_num)*16(out_dim)]
    """
    def __init__(self,in_features,out_features,in_dim,out_dim):
        """
        """
        super(CapsNet,self).__init__()
        #number of output features,10
        self.out_features = out_features
        #number of input features,1152
        self.in_features = in_features
        #dimension of input capsule
        self.in_dim = in_dim
        #dimension of output capsule
        self.out_dim = out_dim
        
        #full connect parameter W with shape [1(batch共享),1152,10,8,16]
        self.W = nn.Parameter(torch.randn(1,self.in_features,self.out_features,in_dim,out_dim))
        
    def forward(self,x):
        #input x,shape=[batch_size,in_features,in_dim]
        #[batch_size,1152,8]
        # (batch, input_features, in_dim) -> (batch, in_features, out_features,1,in_dim)
        x = torch.stack([x] * self.out_features, dim=2).unsqueeze(3)
        
        W = torch.cat([self.W] * conf.batch_size,dim=0)
        # u_hat shape->(batch_size,in_features,out_features,out_dim)=(batch,1152,10,1,16)
        u_hat = torch.matmul(x,W)
        #b for generate weight c,with shape->[1,1152,10,1]
        b = torch.zeros([1,self.in_features,self.out_features,1]).double()
        if self.cuda:
            b = b.cuda()
        b = Variable(b)
        for i in range(3):
            c = F.softmax(b,dim=2)
            #c shape->[batch_size,1152,10,1,1]
            c = torch.cat([c] * conf.batch_size, dim=0).unsqueeze(dim=4)
            #s shape->[batch_size,1,10,1,16]
            s = (u_hat * c).sum(dim=1,keepdim=True)
            #output shape->[batch_size,1,10,1,16]
            v = utils.squash(s,dim=-1)
            v_1 = torch.cat([v] * self.in_features, dim=1)
            #(batch,1152,10,1,16)matmul(batch,1152,10,16,1)->(batch,1152,10,1,1)
            #squeeze
            #mean->(1,1152,10,1)
            #print u_hat.shape,v_1.shape
            update_b = torch.matmul(u_hat,v_1.transpose(3,4)).squeeze(dim=4).mean(dim=0,keepdim=True)
            b = b+update_b
        return v.squeeze(1).transpose(2,3)
```

其他的常规代码可以参看我的github地址:https://github.com/selous123/pytorch-example/tree/master/capsulenet

