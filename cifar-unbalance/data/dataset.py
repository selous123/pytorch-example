#coding:utf-8
ROOT_PATH = "/mnt/hgfs/ubuntu14/dataset/cifar-unbalance-data/"
import torch
import torch.utils.data as data
import cPickle as pickle
import numpy as np
import os

##read the whole file
class cifarUnbalanceDataset(data.Dataset):
    """
    init dataset class
    Args:
        root : root path
        train: is trainging or test
        
    """
    def __init__(self,root,train):
        self.root = root
        self.train = train
        if self.train:
            ##构建路径
            train_path = "train.pk1"
            path = os.path.join(self.root,train_path)
            ##读取文件内容
            f = open(path,"rb")
            datadict = pickle.load(f)
            self.train_data = datadict["data"]
            self.train_labels = datadict["labels"]
            #upsample
            fake_data = self.train_data[self.train_labels.squeeze()==1]
            fake_labels = np.ones(shape=(fake_data.shape[0],1))
            for i in range(99):
                self.train_data = np.concatenate((self.train_data,fake_data),axis=0)
                self.train_labels = np.concatenate((self.train_labels,fake_labels),axis=0)
            
        else:
            test_path = "test.pk1"
            path = os.path.join(self.root,test_path)
            ##读取文件内容
            f = open(path,"rb")
            datadict = pickle.load(f)
            self.test_data = datadict["data"]
            self.test_labels = datadict["labels"]
    
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
if __name__=='__main__':
    train_dataset = cifarUnbalanceDataset(ROOT_PATH,train=True)
    dataloader = data.DataLoader(train_dataset,batch_size=128,shuffle=True,drop_last = True)
    dataiter = iter(dataloader)
    images,labels = dataiter.next()
# =============================================================================
#     for epoch in range(10):
#         print "epoch:{}".format(epoch)
#         for images,labels in dataiter:
#             print "hello world"
# =============================================================================
    
    print images.shape
    print np.sum(labels.numpy()==1)