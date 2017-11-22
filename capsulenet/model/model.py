#coding:utf-8
import torch.nn as nn
import torch.nn.functional as F
import capsconv
import capsnet
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,256,9)
        self.caps_conv = capsconv.CapsConv(256,8)
        self.caps_net = capsnet.CapsNet(1152,10,8,16)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.caps_conv(x)
        x = self.caps_net(x)
        #print x.shape
        return x
    

if __name__=="__main__":
    net = Net()
    for name,parameter in net.named_parameters():
        print name,parameter.shape
