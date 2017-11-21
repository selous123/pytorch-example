#coding:utf-8
import torch
from torch.autograd.variable import Variable
import config
conf = config.DefaultConf()
def squash(x,dim):
    #we should do spuash in each capsule.
    #we use dim to select
    sum_sq = torch.sum(x**2,dim=dim,keepdim=True)
    sum_sqrt = torch.sqrt(sum_sq)
    return (sum_sq/(1.0+sum_sq))* x/sum_sqrt
    
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
