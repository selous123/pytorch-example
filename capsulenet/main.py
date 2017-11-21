#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from model.model import Net
import config
import torch
import torch.optim as optim
from data.dataset import mnistData
from torch.autograd import Variable
import utils
conf = config.DefaultConf()
"""
Created on Fri Nov 17 16:36:52 2017

@author: lrh
"""

def train(net):
    ###read file
    train_dataset = mnistData(conf.root_path,train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=conf.batch_size,shuffle=True,drop_last=True)
    #dataiter = iter(dataloader)
    
    ###optimizer
    #optimize = optim.SGD(net.parameters(),lr = conf.lr)
    optimize = optim.SGD(net.parameters(),lr = conf.lr)
    for name,parameter in net.named_parameters():
        print name,parameter.shape
    raw_input("wait")
    for epoch in range(conf.epoch_num):
        for data in enumerate(dataloader,0):
            images,labels = data
            if conf.cuda:
                images,labels = images.cuda(),labels.cuda()
            #print labels.type
            images,labels= Variable(images),Variable(labels)
            
            v = net(images)
            l = utils.loss(labels,v)
            optimize.zero_grad()
            l.backward()
            optimize.step()
        print "epoch is {},loss is {}".format(epoch,l.data[0])

def test(net):
    pass

if __name__=="__main__":
    net = Net()
    if conf.cuda:
        net.cuda()
    net.double()
    if conf.istraining:
        train(net)
        torch.save(net.state_dict(),"init.pkl")
    else:
        net.load_state_dict(torch.load("init.pkl",map_location=lambda storage, loc: storage))
        test(net)
    #print "learning rate is {}".format(conf.lr)