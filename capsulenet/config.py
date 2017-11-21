#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:03:09 2017

@author: lrh
"""

class DefaultConf(object):
    def __init__(self):
        self.batch_size = 4
        self.root_path = "/mnt/hgfs/ubuntu14/dataset/mnist"
        self.istraining = True
        self.cuda= False
        self.lr = 0.001
        
        self.epoch_num = 500
    
if __name__=="__main__":
    conf = DefaultConf()
    print conf.batch_size