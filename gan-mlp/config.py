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
        #maybe value 'mlp-gan'[0]、'dc-gan'[1]
        self.method = 1
        ##net configuration
        self.lr = 0.00005
        self.beta = (0.5,0.999)
        ##system configuration
        self.root_path = "/home/lrh/dataset/mnist"
        #batch size
        self.batch_size = 64
        #istraining
        self.istraining = True

        self.epoch_num = 20
        #cuda or not
        self.cuda = True
        self.bn = True
        #store pkl name
        self.d_steps = 5
        self.g_steps = 1
        self.result_directory = "result_wdcgan/"

if __name__=="__main__":
    conf = DefaultConf()
    print conf.batch_size
