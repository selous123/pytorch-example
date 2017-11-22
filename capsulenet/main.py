#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from model.model import Net
import config
import torch
import torch.optim as optim
from data.dataset import mnistData
from torch.autograd import Variable
import utils
from visualize.visutils import visualize_loss
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
    if conf.debug:
        for name,parameter in net.named_parameters():
            print name,parameter.shape
        raw_input("wait")
    for epoch in range(conf.epoch_num):
        for i,data in enumerate(dataloader,0):
            images,labels = data
            if conf.cuda:
                images,labels = images.cuda(),labels.cuda()
            #print labels.type
            images,labels= Variable(images),Variable(labels)
            
            v = net(images)
            l = utils.loss(labels,v)
            if conf.visualize:
                conf.train_loss_win=visualize_loss(epoch*len(dataloader)+i,l.data.cpu(),conf.train_loss_env,conf.train_loss_win)
            optimize.zero_grad()
            l.backward()
            optimize.step()
            print "step is {},loss is {}".format(i,l.data[0])
        print "epoch is {},loss is {}".format(epoch,l.data[0])

def test(net):
    test_dataset = mnistData(conf.root_path,train=False)
    dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=conf.batch_size,shuffer=False,drop_last=False)
    predicted_true_num = 0
    total_num = 0
    
    for i,data in enumerate(dataloader,0):
        #labels shape[batch_size,]
        images,labels = data
        if conf.cuda:
            images = images.cuda()
        images= Variable(images)
        #shape->[batch_size,10,16,1]
        v = net(images)
        #shape->[batch_size,10]
        v_norm = torch.sqrt(torch.sum(v**2,dim=2,keepdim=True)).squeeze()
        
        #shape->[batch_size,]
        _,predicted = v_norm.max(dim=1)
        
        predicted_true_num += torch.sum(predicted==labels)
        total_num += labels.shape[0]
    test_acc = predicted_true_num/total_num
    print "accuracy of test is {}".format(test_acc)
    

if __name__=="__main__":
    net = Net()
    if conf.cuda:
        net.cuda()
    net.double()
    if conf.istraining:
        train(net)
        torch.save(net.state_dict(),"pkls/mnist_capsule.pkl")
    else:
        net.load_state_dict(torch.load("pkls/mnist_capsule.pkl",map_location=lambda storage, loc: storage))
        test(net)
    #print "learning rate is {}".format(conf.lr)
