#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:03:09 2017

@author: lrh
"""

class DefaultConf(object):
    def __init__(self):
        #print debug information
        self.debug = True
        ##net configuration

        #self.lr = 0.0002
        self.lr = 0.001
        self.beta = (0.5,0.999)
        ##system configuration
        self.root_path = "/home/lrh/dataset/mnist"
        #batch size
        self.batch_size = 32
        #istraining
        self.istraining = True
        self.fixz = False
        self.epoch_num = 200
        #cuda or not
        self.cuda = True
        self.bn = True
        #store pkl name
        self.g_steps = 5
        self.result_directory = "mnist_0913"
if __name__=="__main__":
    conf = DefaultConf()
    print conf.batch_size
