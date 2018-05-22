#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from model.model import Net
import config
conf = config.DefaultConf()
import torch
import torch.optim as optim
from data.dataset import mnistData
from torch.autograd import Variable

"""
Created on Fri Nov 17 16:36:52 2017

@author: lrh
"""
import numpy as np
from tabulate import tabulate
from visualize.visutils import visualize_loss


def loss(logits,labels,*args):
    """
    Args:
        logits: prediction of samples,[batch_size,10]
        labels: true label of samples,[batch_size,1]
    Return:
        cross entropy loss,[1,]
    """
    loss = -torch.mean(labels*torch.log(logits))
    return loss

def train(net):

    ###read file
    train_dataset = mnistData(conf.root_path,train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=conf.batch_size,shuffle=True,drop_last=True)
    #dataiter = iter(dataloader)

    ###optimizer
    optimize = optim.SGD(net.parameters(),lr = conf.lr)

    for epoch in range(conf.epoch_num):
        #set module.istraing=True
        #This has any effect only on modules such as Dropout or BatchNorm.
        net.train()
        running_loss = []
        for i,data in enumerate(dataloader,0):
            images,labels = data
            images,labels = images.type(torch.DoubleTensor),labels.type(torch.DoubleTensor)
            if conf.cuda:
                images,labels = images.cuda(),labels.cuda()
            #print type(images)
            images,labels = Variable(images),Variable(labels)
            #print images[1]
            #print labels.data.numpy()
            #raw_input("wait")
            logits = net(images)
            #print logits.data.view(64,8)
            #choose "easy" sample
            l = loss(logits,labels)
            running_loss.append(l.data.cpu().numpy())
            optimize.zero_grad()
            #visulize loss
            if conf.visualize:
                conf.train_loss_win=visualize_loss(epoch*len(dataloader)+i,l.data.cpu(),conf.train_loss_env,conf.train_loss_win)
            l.backward()
            optimize.step()
            #print "epoch is:{},step is:{},loss is:{}".format(epoch,i,l.data[0])
        print "epoch is:{},loss is:{}".format(epoch,l.item())



def test(net):

    test_dataset = mnistData(conf.root_path,train=False)
    data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=512,shuffle=False,drop_last=False)
    num_examples = 0
    predicted_num = 0
    for images,labels in data_loader:
        #move the images to GPU
        images = images.type(torch.DoubleTensor)
        if conf.cuda:
            images = images.cuda()
        logits = net(Variable(images))
        ##greater and equal
        #print logits
        _,predicted = logits.data.max(dim=1)
        #print predicted
        if conf.cuda:
            predicted = predicted.cpu()

        predicted = predicted.numpy()
        #print predicted
        labels = labels.numpy()
        num_examples+=labels.shape[0]
        predicted_num += np.sum(predicted == labels)
    test_acc = predicted_num/num_examples
    print "accuracy of test data is {}".format(test_acc)
# =============================================================================
#         print predicted.size()
#         print labels.squeeze().size()
#         raw_input("wait")
#
#        #labels = labels.type('torch.ByteTensor')
#        total += labels.size(0)
#        correct += (predicted.cpu().numpy() == labels.numpy()).sum()
# =============================================================================
if __name__=='__main__':


    net = Net()
    if conf.cuda:
        net.cuda()
    net.double()
    if conf.istraining:
        #set moudle.istraing=True
        net.train()
        train(net)
        torch.save(net.state_dict(),conf.pkl_name)
    else:
        net.load_state_dict(torch.load(conf.pkl_name,map_location=lambda storage, loc: storage))
        #set module.istraing=False
        #This has any effect only on modules such as Dropout or BatchNorm.
        net.eval()
        test(net)
    #print "learning rate is {}".format(conf.lr)
