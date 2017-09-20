#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:02:48 2017

@author: lrh
"""


import torch
import torchvision
#%%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        #(28-5+1)/2=12
        self.conv2 = nn.Conv2d(6, 16, 5)
        #(12-5+1)/2=4
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        print "after conv1 size is {}".format(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print "after conv2 size is {}".format(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

net.zero_grad()



target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss.grad_fn)

#loss size:(1,)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


##update parameters
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

running_loss = 0.0
# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

print type(loss)
print type(loss.data[0])

running_loss += loss.data[0]
print type(running_loss)
print('loss: %.3f' %(running_loss / 2000))

#%%
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2),requires_grad=True)

y = x*3


print y

y.backward(2*torch.ones(2,2))


print x.grad
#%%
import torch
from torch.autograd import Variable

x = torch.randn(3)

print "x type:{}".format(x)
x = Variable(x, requires_grad=True)


y = x.mul(2)
out = y

print "y type is {},,,,".format(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
out.backward()

print x.grad
#%%
class F(object):
    def __call__(self,input):
        return self.forward(input)
    def forward(self,input):
        return 2*input


class Test(object):
    def __init__(self):
        self.f = F()
    def forward_test(self):
        return self.f(2)



t = Test()
print t.forward_test()


#%%
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.Tensor([1,2]))


############################################module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.weight = nn.Parameter(torch.Tensor([[1,2]]))
        self.bias = nn.Parameter(torch.Tensor([[1]]))
        
    def forward(self,x):
        x = torch.sigmoid(self.weight.mm(x)+self.bias)
        return x
    
net = Net()
print(net)
#input = Variable(torch.Tensor(1,1,32,32))

input = Variable(torch.Tensor([[1],[2]]))

#for parameter in net.parameters():
#    print parameter

labels = 1;

##############################################loss
def loss_fn(labels,logits):
    l = labels*torch.log(logits)+(1-labels)*(torch.log(1-logits))
    return -l;




###############################################optimizer
optimizer = optim.SGD(net.parameters(),lr = 0.01, momentum=0.9)


running_loss = 0.0
############################################training
for i in xrange(1000):
    logits = net(input)
    loss = loss_fn(labels,logits)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        print "loss is:{}".format(loss.data[0][0])
        #print loss.data.numpy()
        
print(net(input).data[0][0])


















