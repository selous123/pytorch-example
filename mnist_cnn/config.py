#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:03:09 2017

@author: lrh
"""

class DefaultConf(object):
    def __init__(self):
        #print debug information
        self.debug = False
        
        ##net configuration
        self.lr = 0.01
        ##system configuration
        self.root_path = "/home/lrh/dataset/mnist"
        #batch size
        self.batch_size = 1024
        #istraining
        self.istraining = False
        #cuda or not
        self.cuda = True
        
        ##batch normalization
        self.bn=True
        
        #store pkl name
        self.pkl_name = "pkls/mnist_init_300epoch.pkl"
        self.epoch_num = 300
        
        self.visualize = False
        #visualize train loss
        self.train_loss_env = "mnist_train20171121"
        self.train_loss_win = None
    
if __name__=="__main__":
    conf = DefaultConf()
    print conf.batch_size
