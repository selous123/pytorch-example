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
        b = Variable(torch.zeros([1,self.in_features,self.out_features,1]).double())
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
    

if __name__ == "__main__":
    net = CapsNet()
    print net