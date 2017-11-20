#coding:utf-8

import torch.utils.data as data
import input_mnist_data as input_data
import sys
##read the whole file
class mnistData(data.Dataset):
    """
    init dataset class
    Args  :
        root : root path
        train: is trainging or test
    Return:
        self.root,self.train,self.data,self.labels
        
    """
    def __init__(self,root,train):
        self.root = root
        self.train = train
        #train dataset
        if self.train:
            self.train_data,self.train_labels = input_data.read_train_data(self.root)
        #test dataset
        else:
            self.test_data,self.test_labels = input_data.read_test_data(self.root)

    
    def __getitem__(self,index):
        if self.train:
            return self.train_data[index],self.train_labels[index]
        else:
            return self.test_data[index],self.test_labels[index]
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
