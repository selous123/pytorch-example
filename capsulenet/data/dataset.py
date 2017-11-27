#coding:utf-8

import torch.utils.data as data
import input_mnist_data as input_data
import numpy as np


def dense_to_one_hot(labels,class_num):
    
    height = labels.shape[0]
    a = np.zeros(shape=[height,class_num])
    a[np.arange(height),labels] = 1
    return a

def augmentation(x, max_shift=2):
    _, _, height, width = x.shape

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = np.zeros(x.shape)
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image
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
    def __init__(self,root,train,augment=True):
        self.root = root
        self.train = train
        #train dataset
        if self.train:
            self.train_data,self.train_labels = input_data.read_train_data(self.root)
            self.train_labels = dense_to_one_hot(self.train_labels,10)
            if augment:
                self.train_data = augmentation(self.train_data / 255.0)
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
        
