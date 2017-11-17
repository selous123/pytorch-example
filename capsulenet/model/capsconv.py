#coding:utf-8
#author:selous
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils


class CapsConv(nn.Module):
    def __init__(self,in_channels=256,out_dim=8):
        self.in_channels = in_channels
        self.out_dim = out_dim
        super(CapsConv,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=32,  # fixme constant
                              kernel_size=9,  # fixme constant
                              stride=2, # fixme constant
                              bias=True)
        
    def forward(self,x):
        #input x with shape ->[batch_size,in_features,height,width]
        #output with shape->[batch_size,32,6,6]
        x = [self.conv(x) for i in range(self.out_dim)]
        #output with shape->[batch_size,8,32,6,6]
        x = torch.stack(x,dim=1)
        #return shape->[batch_size,1152,8]
        x = utils.squash(x,dim=2)
        return x.view(x.size(0),self.out_dim,-1).transpose(1,2)