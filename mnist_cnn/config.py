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
        self.lr = 0.001
        ##system configuration
        self.root_path = "/home/lili/Tao_Zhang/dataset/mnist"
        #batch size
        self.batch_size = 1024
        #istraining
        self.istraining = True
        #cuda or not
        self.cuda = True
        
        ##batch normalization
        self.bn=True
        
        #store pkl name
        self.pkl_name = "mnist_init.pkl"
        self.epoch_num = 100
        
        self.visualize = True
        #visualize train loss
        self.train_loss_env = "malware_train20171115"
        self.train_loss_win = None
    
if __name__=="__main__":
    conf = DefaultConf()
    print conf.batch_size
